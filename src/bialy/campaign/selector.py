"""The selector stage: three decision heads fitted on the campaign's shared splits and
frozen as P0.

The existing recipes train each head on `train`; the host's calibration sets thresholds
on `validation`; the host's replay measures the heads on `development` against the bars
in config/campaign/selector.yaml. `system_test` is never read. P0 binds the heads, the
backbone, the catalog, the decision contract and every threshold, and is whole or absent.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from . import ancestry
from .execution import atomic_json
from .journal import CampaignFault, digest

CONFIG = Path(__file__).resolve().parents[3] / 'config/campaign/selector.yaml'
SCHEMA = 'pw-selector-policy/1'
HEADS = ('turn-load', 'unit-rank', 'code-rank')
COMMIT = re.compile(r'[0-9a-f]{40}')


def load(path=CONFIG):
    value = yaml.safe_load(Path(path).read_text())
    if set(value['heads']) != set(HEADS) or not 0 < value['fallback_rate'] < 1 or set(value['bars']) != set(HEADS):
        raise CampaignFault('selector_configuration_invalid')
    return value


def measure(report, path):
    for key in path.split('.'):
        if not isinstance(report, dict) or key not in report:
            return None
        report = report[key]
    return report


def failures(bars, measures):
    """Every bar a head falls short of, with the measured value."""
    out = []
    for head, rules in bars.items():
        for rule in rules:
            value = measure(measures[head], rule['metric'])
            if not isinstance(value, (int, float)) or value < rule['at_least']:
                out.append({'head': head, 'metric': rule['metric'], 'value': value, 'at_least': rule['at_least']})
    return out


# Where the exported catalog carries each decision section; replay reads thresholds there.
CATALOG_SECTIONS = {'turn': 'questions', 'request': 'request', 'lookup': 'lookup'}


def catalog_path(key):
    section, rest = key.split('.', 1)
    if section not in CATALOG_SECTIONS:
        return None
    return CATALOG_SECTIONS[section] + '.' + rest


def thresholds(config, decisions, calibrated, catalog):
    """decisions.yaml's thresholds, with those the host's calibration printed replacing its
    calibrated keys. A key calibration leaves unprinted keeps the contract's value, as the
    calibration itself states. The catalog must agree with the contract it was exported with."""
    out, tuned = {}, []
    for kind in ('bound', 'calibrated'):
        for section, keys in config[kind].items():
            for key in keys:
                name = section + '.' + key
                out[name] = measure(decisions, name)
                exported = measure(catalog, catalog_path(name)) if catalog_path(name) else None
                if exported is not None and exported != out[name]:
                    raise CampaignFault('selector_catalog_contract_mismatch', details={'threshold': name})
                if kind == 'calibrated' and measure(calibrated, name) is not None:
                    out[name] = measure(calibrated, name)
                    tuned.append(name)
    missing = sorted(name for name, value in out.items() if value is None)
    if missing:
        raise CampaignFault('selector_thresholds_missing', details={'thresholds': missing})
    return dict(sorted(out.items())), sorted(tuned)


def calibrated_catalog(catalog, values, path):
    """The catalog with P0's thresholds in place, the form the host's replay reads them in."""
    document = json.loads(json.dumps(catalog))
    for name, value in values.items():
        target = catalog_path(name)
        if target is None:
            continue
        *parents, leaf = target.split('.')
        node = document
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    atomic_json(path, document)
    return Path(path)


def stratum(selector, task_id, repetition):
    """Which selector a trial runs under: P0, or a planned fallback to the conservative reference
    drawn at P0's fallback rate. A conservative selection has no strata."""
    if not selector.get('pilot'):
        return 'conservative'
    draw = int(digest({'policy': selector['policy'], 'task': task_id, 'repetition': repetition})[:12], 16) / float(1 << 48)
    return 'planned_fallback' if draw < selector['fallback_rate'] else 'p0'


def plan_record(selector):
    """The selection the evaluation plan freezes."""
    return {'policy': selector.get('policy'), 'fallback_rate': selector.get('fallback_rate', 0) if selector.get('pilot') else 0}


def partitioned_dumps(directory, assignment):
    """Code-rank dumps by their repository's partition; a dump outside the shared graph is not used."""
    out = {}
    for dump in sorted(Path(directory).glob('*.jsonl')):
        repository = dump.stem.split('-', 2)[-1]
        try:
            partition = assignment.partition('repository:' + repository)
        except CampaignFault:
            continue
        out.setdefault(partition, []).append(dump)
    return out


class Workbench:
    """The existing trainers and the host's calibration and replay, over one head-row run."""

    def __init__(self, run, process=subprocess.run):
        self.run, self.process = run, process

    def inputs(self, split, dumps):
        from .. import training
        judged = self.run.path('skillreq', 'judged.jsonl')
        if not judged.exists():
            raise CampaignFault('selector_unqualified', details={'failures': [{'head': 'unit-rank', 'metric': 'skill_requests', 'value': None}]})
        corpus = self.catalog()
        return {'corpus': corpus, 'train': split / 'train.jsonl', 'val': split / 'val.jsonl',
                'independent_corpus': training.independent_corpus(self.run.checkout, corpus, self.run.path('independent-corpus.json')),
                'skillreq_train': training.skillreq_partition(judged, 'train', self.run.path('skillreq', 'train.jsonl')),
                'skillreq_eval': training.skillreq_partition(judged, 'validation', self.run.path('skillreq', 'validation.jsonl')),
                'dumps_train': dumps.get('train', []), 'dumps_val': dumps.get('validation', [])}

    def train(self, recipe, inputs, heads_dir):
        from .. import training
        settings = dict(self.run.setting('train', {}), retention=self.run.setting('retention', {}))
        return training.train(recipe, inputs, self.run.checkout, self.run.factory.root, settings, heads_dir,
                              '%s-selector-%s' % (self.run.name, recipe), float(settings.get('max_hours', 24)) * 3600)

    def launcher(self, heads, path):
        from .. import build
        from ..run import model_dir
        return build.launcher(self.run.path('build', 'host-engine', 'bialy'), model_dir(self.run), heads, path)

    def host(self, script, corpus, arguments):
        from .. import training
        result = self.process([sys.executable, 'scripts/bialy/' + script, '--corpus', str(corpus), *arguments],
                              cwd=str(self.run.checkout), capture_output=True, text=True,
                              env=training.environment(self.run.factory.root, self.run.setting('train', {})))
        if result.returncode:
            raise CampaignFault('selector_host_script_failed', details={'script': script, 'stderr': result.stderr[-2000:]})
        return result.stdout

    def calibrate(self, launcher, rows):
        return yaml.safe_load(self.host('calibrate.py', self.catalog(), ['--examples', str(rows), '--engine', str(launcher)])) or {}

    def replay(self, launcher, corpus, rows, out):
        self.host('replay_eval.py', corpus, ['--examples', str(rows), '--engine', str(launcher), '--json', str(out)])
        return json.loads(Path(out).read_text())

    def pilot(self, heads_dir, values, out):
        """The runner mount: engine, checkpoint, heads, and the image's config with P0's thresholds."""
        from .. import build, fleet
        from ..run import model_dir
        engine = 'host-engine' if build.host_is_runner_platform() else 'runner-engine'
        build.pilot(self.run.path('build', engine, 'bialy'), model_dir(self.run), heads_dir, out)
        fleet.stage_config_overlay(Path(self.run.checkout) / 'lycaon/config', out, self.run.setting('engine_on', {}).get('deadline_ms'),
                                   self.run.collection_window, values)
        return Path(out)

    def decisions(self, config):
        return yaml.safe_load((Path(self.run.checkout) / config['decisions']).read_text())

    def catalog(self):
        return Path(self.run.path('corpus.json'))

    def backbone(self):
        from .. import training
        return training.backbone(self.run.factory.root)


def file_digest(path):
    from ..provenance import file_sha256
    return file_sha256(path)


def tree_digest(root):
    root = Path(root)
    return digest({str(item.relative_to(root)): file_digest(item) for item in sorted(root.rglob('*')) if item.is_file()})


class SelectorTraining:
    """Train the three heads on the shared train split and freeze P0 from validation thresholds."""

    def __init__(self, journal, run, workbench=None, config=None):
        self.journal, self.run = journal, run
        self.workbench, self.config = workbench or Workbench(run), config or load()
        self.directory = journal.directory / 'selector'

    def split(self, rows):
        assignment = ancestry.load(self.run.path('ancestry.json'))
        if assignment.sha256 != rows['assignment_sha256']:
            raise CampaignFault('selector_split_not_shared')
        split = self.run.path('split')
        missing = [name for name in ('train.jsonl', 'val.jsonl', 'development.jsonl') if not (split / name).exists()]
        if missing:
            raise CampaignFault('selector_split_missing', details={'files': missing})
        return assignment, split

    def execute(self, inputs, attempt):
        assignment, split = self.split(inputs['rows'])
        dumps = partitioned_dumps(self.run.path('coderank', 'dumps'), assignment)
        if not dumps.get('train') or not dumps.get('validation'):
            raise CampaignFault('selector_unqualified', details={'failures': [{'head': 'code-rank', 'metric': 'dumps', 'value': None}]})
        training_inputs = self.workbench.inputs(split, dumps)
        heads = {}
        for head, recipe in self.config['heads'].items():
            result = self.workbench.train(recipe, training_inputs, self.directory / 'heads')
            if 'head' not in result:
                raise CampaignFault('selector_unqualified', details={'failures': [{'head': head, 'metric': 'trained', 'value': result.get('skipped')}]})
            heads[head] = result['head']
        launcher = self.workbench.launcher(heads, self.directory / 'launcher.sh')
        catalog = json.loads(self.workbench.catalog().read_text())
        values, tuned = thresholds(self.config, self.workbench.decisions(self.config),
                                   self.workbench.calibrate(launcher, split / 'val.jsonl'), catalog)
        policy = {'schema': SCHEMA, 'heads': {head: {'recipe': self.config['heads'][head], 'sha256': file_digest(path)}
                                              for head, path in sorted(heads.items())},
                  'backbone_sha256': tree_digest(self.workbench.backbone()), 'catalog_sha256': file_digest(self.workbench.catalog()),
                  'decision_contract': self.workbench.decisions(self.config)['version'], 'thresholds': values, 'calibrated': tuned,
                  'fallback_rate': self.config['fallback_rate'], 'assignment_sha256': assignment.sha256}
        policy['identity'] = digest(policy)
        atomic_json(self.directory / 'p0.json', policy)
        corpus = calibrated_catalog(catalog, values, self.directory / 'corpus.json')
        pilot = self.workbench.pilot(self.directory / 'heads', values, self.directory / 'pilot')
        return {'policy': policy, 'heads': heads, 'launcher': str(launcher), 'corpus': str(corpus), 'pilot': str(pilot)}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        policy = json.loads((self.directory / 'p0.json').read_text())
        if (policy != output['policy'] or digest({k: v for k, v in policy.items() if k != 'identity'}) != policy['identity']
                or any(file_digest(path) != policy['heads'][head]['sha256'] for head, path in output['heads'].items())):
            raise CampaignFault('selector_policy_changed')


class SelectorQualification:
    """Qualify the three heads together on development; P0 is whole or absent."""

    def __init__(self, journal, run, workbench=None, config=None):
        self.journal, self.run = journal, run
        self.workbench, self.config = workbench or Workbench(run), config or load()

    def execute(self, inputs, attempt):
        trained = inputs['selector']
        self.workbench.replay(trained['launcher'], trained['corpus'], self.run.path('split', 'development.jsonl'), self.report())
        return self.judge(trained)

    def report(self):
        return self.journal.directory / 'selector' / 'development.json'

    def judge(self, trained):
        replay = json.loads(self.report().read_text())
        code_rank = json.loads(Path(trained['heads']['code-rank']).with_suffix('.metrics.json').read_text())
        missing = failures(self.config['bars'], {'turn-load': replay, 'unit-rank': replay, 'code-rank': code_rank})
        if missing:
            raise CampaignFault('selector_unqualified', details={'failures': missing})
        policy = trained['policy']
        return {'qualified': True, 'policy': policy['identity'],
                'configuration': {'pilot': trained['pilot'], 'policy': policy['identity'],
                                  'thresholds': policy['thresholds'], 'fallback_rate': policy['fallback_rate']}}

    def reconcile(self, inputs, attempt):
        if self.report().exists():
            return self.judge(inputs['selector'])
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output != self.judge(inputs['selector']):
            raise CampaignFault('selector_qualification_changed')
