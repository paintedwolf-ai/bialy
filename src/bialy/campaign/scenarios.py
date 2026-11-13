"""The configured scenario bank: eight behavior focuses, groups of parameter variants,
deterministic graders outside the workspace, and ancestry-allocated split pools.

A group is one situation; its variants differ only in parameters and always share
a partition. Graders run against a copy of the delivered source with the host
bundle beside them (`oracle.grade`); a scenario prompt never names a host tool.
"""
import shutil
import string
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..provenance import file_sha256
from . import ancestry
from .execution import atomic_json
from .journal import CampaignFault, digest

CATALOG = Path(__file__).resolve().parents[3] / 'config/campaign/scenarios.yaml'
SUPPORT = Path(__file__).resolve().with_name('grader_support.py')
GROUP_KEYS = {'id', 'focus', 'workspace', 'prompt', 'fixtures', 'grader', 'verification', 'solution', 'superficial',
              'tampered', 'follow_ups', 'diagnostic', 'variants', 'source_class'}
REQUIRED = {'id', 'focus', 'workspace', 'prompt', 'grader', 'solution', 'variants'}
# Expectations a group may register; BF06 admits the expected failing turn only with the witness check.
DIAGNOSTICS = {'reproduced_failure'}
DIAGNOSTIC_PROGRAM = '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grader_support import reproduced_failure
assert reproduced_failure(), 'no failed verification of the original source preceded the change'
'''


@dataclass(frozen=True)
class Group:
    id: str
    focus: str
    workspace: str
    prompt: str
    grader: str
    solution: dict
    variants: tuple
    fixtures: dict = field(default_factory=dict)
    verification: bool = True
    superficial: dict | None = None
    tampered: dict | None = None
    follow_ups: tuple = ()
    diagnostic: dict | None = None
    source_class: str = 'synthetic'

    def render(self, template, parameters):
        try:
            return string.Template(template).substitute(parameters)
        except (KeyError, ValueError) as exc:
            raise CampaignFault('scenario_template_invalid', details={'group': self.id, 'error': str(exc)}) from exc

    def files(self, files, parameters):
        return {name: self.render(body, parameters) for name, body in (files or {}).items()}


@dataclass(frozen=True)
class Catalog:
    focuses: tuple
    groups: tuple
    quotas: dict

    def by_focus(self):
        out = {focus: [] for focus in self.focuses}
        for group in self.groups:
            out[group.focus].append(group)
        return out


def load(path=CATALOG):
    value = yaml.safe_load(Path(path).read_text())
    focuses = tuple(item['id'] for item in value['focuses'])
    quotas = value['quotas']
    if len(set(focuses)) != len(focuses):
        raise CampaignFault('scenario_focus_duplicate')
    groups, seen = [], set()
    for raw in value.get('groups') or []:
        missing, unknown = REQUIRED - set(raw), set(raw) - GROUP_KEYS
        if missing or unknown:
            raise CampaignFault('scenario_group_invalid', details={'group': raw.get('id'), 'missing': sorted(missing), 'unknown': sorted(unknown)})
        if (raw['id'] in seen or raw['focus'] not in focuses
                or raw.get('diagnostic') is not None and raw['diagnostic'].get('expectation') not in DIAGNOSTICS):
            raise CampaignFault('scenario_group_invalid', details={'group': raw['id']})
        seen.add(raw['id'])
        variants = tuple(raw['variants'])
        if not variants or len(variants) > quotas['variants_per_group'] or len({digest(v) for v in variants}) != len(variants):
            raise CampaignFault('scenario_variant_quota', details={'group': raw['id'], 'variants': len(variants),
                                                                   'maximum': quotas['variants_per_group']})
        groups.append(Group(id=raw['id'], focus=raw['focus'], workspace=raw['workspace'], prompt=raw['prompt'],
                            grader=raw['grader'], solution=raw['solution'], variants=variants,
                            fixtures=raw.get('fixtures') or {}, verification=raw.get('verification', True),
                            superficial=raw.get('superficial'), tampered=raw.get('tampered'),
                            follow_ups=tuple(raw.get('follow_ups') or ()), diagnostic=raw.get('diagnostic'),
                            source_class=raw.get('source_class', 'synthetic')))
    return Catalog(focuses=focuses, groups=tuple(groups), quotas=quotas)


def grader_program(group, parameters):
    """The grader a task runs: host evidence support, the group's checks, then verification."""
    body = group.render(group.grader, parameters)
    # A script's import path holds its own directory, so the delivered source is added explicitly.
    lines = ['import json, sys', 'from pathlib import Path', 'SOURCE = Path.cwd()', 'sys.path.insert(0, str(SOURCE))',
             'sys.path.insert(0, str(Path(__file__).resolve().parent))',
             'from grader_support import invoked, operations, verified', '', body.rstrip(), '']
    if group.verification:
        lines.append("assert verified(), 'the delivered source has no passed host verification'")
    return '\n'.join(lines) + '\n'


def write_exact(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != body:
        raise CampaignFault('scenario_bank_changed', details={'path': str(path)})
    path.write_text(body)


def tasks(catalog, directory, workspace_path):
    """One task per variant. Its oracle holds the fixtures and the grader, never the agent's workspace."""
    directory = Path(directory)
    out = []
    for group in catalog.groups:
        for index, parameters in enumerate(group.variants):
            identifier = '%s-%d' % (group.id, index)
            root = directory / identifier
            fixtures = group.files(group.fixtures, parameters)
            for name, body in fixtures.items():
                write_exact(root / 'fixtures' / name, body)
            write_exact(root / 'check.py', grader_program(group, parameters))
            if group.diagnostic:
                write_exact(root / 'diagnostic.py', DIAGNOSTIC_PROGRAM)
            support = root / 'grader_support.py'
            if not support.exists():
                shutil.copyfile(SUPPORT, support)
            elif support.read_bytes() != SUPPORT.read_bytes():
                raise CampaignFault('scenario_bank_changed', details={'path': str(support)})
            task = {'id': identifier, 'group': group.id, 'focus': group.focus, 'source_class': group.source_class,
                    'workspace': group.workspace, 'project_path': workspace_path(group.workspace),
                    'prompt': group.render(group.prompt, parameters), 'oracle_directory': identifier,
                    'oracle_files': {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob('*')) if path.is_file()},
                    'fixture_files': {name: 'fixtures/' + name for name in fixtures},
                    'allowed_write_roots': ['.'], 'artifact_paths': [],
                    'checks': [{'id': 'grader', 'argv': ['python3', '/oracle/check.py']}],
                    'required_checks': ['grader', 'host-closeout'], 'prompt_timeout': '10m'}
            if group.follow_ups:
                task['follow_ups'] = [{'prompt': group.render(text, parameters)} for text in group.follow_ups]
            if group.diagnostic:
                task['checks'].append({'id': 'diagnostic', 'argv': ['python3', '/oracle/diagnostic.py']})
                task['required_checks'] = ['grader', 'diagnostic', 'host-closeout']
                task['diagnostic'] = {'expectation': group.diagnostic['expectation'], 'witness': 'diagnostic'}
            out.append(task)
    atomic_json(directory / 'tasks.json', out)
    return out


def assignment(catalog, task_list, seed, plan):
    """Partitions for every group, by ancestry within each focus. `plan` names a quota in
    the catalog; a focus with too few groups is refused with its exact deficit."""
    graph = ancestry.Graph()
    for group in catalog.groups:
        graph.add(scenario_node(group.id), 'scenario', source_class=group.source_class, stratum=group.focus,
                  hashes={'prompt': digest(group.prompt), 'solution': digest(group.solution)})
    for task in task_list:
        node = ancestry.task_node(task['id'])
        graph.add(node, 'task', source_class=task['source_class'], stratum=task['focus'],
                  hashes={'fixture': digest(task['oracle_files'])})
        graph.link(node, scenario_node(task['group']))
    quota = catalog.quotas[plan]
    return ancestry.allocate(graph, seed, quota['fractions'], quota['minimums'])


def scenario_node(group_id):
    return 'scenario:' + group_id


def deficits(catalog, plan):
    """Groups each focus still needs before `plan` can be allocated."""
    minimums = catalog.quotas[plan]['minimums']
    need = sum(minimums.values())
    return {focus: need - len(groups) for focus, groups in catalog.by_focus().items() if len(groups) < need}
