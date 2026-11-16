"""The selector: three decision heads fitted on a run's shared splits and frozen as one
policy, P0. Every head the factory trains, serves in a runner or releases comes from here.

`fit` trains each head on `train`, selects its epoch on `validation`, has the host's
calibration set the thresholds on `validation`, and freezes the policy. `qualify`
measures it on `development`, rows that neither trained nor selected any head: the
host's parity probe, the host's replay, and the host's rerank measure, against the bars
in config/campaign/selector.yaml. `system_test` is read once, after qualification, by
the held-out comparison. P0 binds the heads, the backbone, the catalog, the option
vocabulary the turn-load head was trained with, the decision contract and every
threshold, and is whole or absent.
"""
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from . import ancestry
from .execution import atomic_json
from .journal import CampaignFault, digest

CONFIG = Path(__file__).resolve().parents[3] / 'config/campaign/selector.yaml'
SCHEMA = 'pw-selector-policy/2'
QUALIFICATION = 'pw-selector-qualification/1'
HEADS = ('turn-load', 'unit-rank', 'code-rank')
COMMIT = re.compile(r'[0-9a-f]{40}')


def load(path=CONFIG):
    value = yaml.safe_load(Path(path).read_text())
    parity = value.get('parity') or {}
    if (set(value['heads']) != set(HEADS) or not 0 < value['fallback_rate'] < 1 or set(value['bars']) != set(HEADS)
            or not parity.get('turns', 0) > 0 or not parity.get('tolerance', 0) > 0):
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


def rerank_measure(reports):
    """The host's rerank reports for one head pooled over their pairs: the rank of each
    request's code unit in the site's text-match order and blended with the head."""
    pairs = sum(report['pairs'] for report in reports)
    if not pairs:
        return {'pairs': 0}

    def pooled(side, key):
        return round(sum(report[side][key] * report['pairs'] for report in reports) / pairs, 4)
    out = {'pairs': pairs, 'lexical': {'mrr': pooled('lexical', 'mrr'), 'hit_1': pooled('lexical', 'hit_1')},
           'blended': {'mrr': pooled('blended', 'mrr'), 'hit_1': pooled('blended', 'hit_1')},
           'improved': sum(report['improved'] for report in reports), 'regressed': sum(report['regressed'] for report in reports)}
    out['mrr_gain'] = round(out['blended']['mrr'] - out['lexical']['mrr'], 4)
    return out


RERANK_WORKERS = 4
RERANK_SOURCES = ('docs', 'model')
PARITY_LINE = re.compile(r'parity: worst gap ([0-9.]+) over (\d+) turns')


class Workbench:
    """The trainers and the host's calibration, replay, parity probe and rerank measure,
    over one run's rows."""

    def __init__(self, run, process=subprocess.run):
        self.run, self.process = run, process

    def inputs(self, split):
        from .. import coderank, training
        judged = self.run.path('skillreq', 'judged.jsonl')
        dumps = coderank.training_dumps(self.run.path('coderank', 'dumps'))
        requests = {partition: training.skillreq_partition(judged, partition, self.run.path('skillreq', partition + '.jsonl'))
                    for partition in ('train', 'validation')} if judged.exists() else {}
        return {'corpus': self.catalog(), 'train': split / 'train.jsonl', 'val': split / 'val.jsonl',
                'skillreq_train': requests.get('train'), 'skillreq_eval': requests.get('validation'),
                'dumps_train': dumps['train'], 'dumps_val': dumps['validation']}

    def settings(self):
        return self.run.setting('train', {})

    def train(self, recipe, inputs, heads_dir, label):
        from .. import training
        settings = self.settings()
        return training.train(recipe, inputs, self.run.checkout, self.run.factory.root, settings, heads_dir,
                              '%s-%s-%s' % (self.run.name, label, recipe), float(settings.get('max_hours', 24)) * 3600)

    def engine(self):
        return self.run.path('build', 'host-engine', 'bialy')

    def launcher(self, heads, path):
        from .. import build
        from ..run import model_dir
        return build.launcher(self.engine(), model_dir(self.run), heads, path, self.catalog())

    def host(self, script, arguments, python=sys.executable, accept=(0,)):
        from .. import host_scripts, training
        result = self.process(host_scripts.command(python, script, arguments), cwd=str(self.run.checkout), capture_output=True, text=True,
                              env=training.environment(self.run.factory.root, self.settings()))
        if result.returncode not in accept:
            raise CampaignFault('selector_host_script_failed', details={'script': script, 'stderr': result.stderr[-2000:]})
        return result

    def calibrate(self, launcher, rows):
        printed = self.host('calibrate.py', ['--corpus', self.catalog(), '--examples', rows, '--engine', launcher]).stdout
        return yaml.safe_load(printed) or {}

    def replay(self, launcher, corpus, rows, out, truth='consensus'):
        self.host('replay_eval.py', ['--corpus', corpus, '--examples', rows, '--engine', launcher, '--tool-truth', truth, '--json', out])
        return json.loads(Path(out).read_text())

    def parity(self, launcher, corpus, head, rows, turns, tolerance):
        """The host's probe of the turn-load head: the trainer's own forward against the
        engine on the same turns. It passes only when the head records the encoding the
        engine serves at and every probability agrees within the tolerance."""
        from .. import training
        python = training.venv_python(self.run.factory.root, self.settings())
        result = self.host('parity_probe.py', ['--model', training.BACKBONE[0], '--head', head, '--examples', rows, '--engine', launcher,
                                               '--corpus', corpus, '--n', turns, '--tolerance', tolerance], python=python, accept=(0, 1, 2))
        lines = [line for line in result.stdout.splitlines() if line.startswith('parity:')]
        compared = PARITY_LINE.search(lines[-1]) if lines else None
        return {'passed': result.returncode == 0, 'tolerance': tolerance, 'summary': lines[-1] if lines else result.stderr[-400:],
                'worst_gap': float(compared.group(1)) if compared else None, 'turns': int(compared.group(2)) if compared else 0}

    def rerank(self, head, partition, out_dir):
        """The code-rank head through the host's rerank evaluator and engine, over every
        site of the repositories the ancestry puts in `partition`."""
        from .. import coderank
        from ..run import model_dir, scaled_factory
        env = dict(os.environ, LYCAON_DECIDE_BINARY=str(self.engine()), LYCAON_DECIDE_MODEL_DIR=str(model_dir(self.run)),
                   LYCAON_DECIDE_HEADS='code-rank=%s' % head, HF_HUB_OFFLINE='1')
        binary = str(self.run.host_bin_dir / 'decide-rerank')
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for repo in coderank.repositories(scaled_factory(self.run), ancestry.load(self.run.path('ancestry.json')))[partition]:
            units = self.run.path('coderank', 'units', repo.name + '.jsonl')
            for source in RERANK_SOURCES:
                pairs = self.run.path('coderank', 'pairs', '%s-%s.jsonl' % (source, repo.name))
                if units.exists() and pairs.exists() and pairs.stat().st_size:
                    jobs += [(site, source, repo, units, pairs) for site in coderank.SITES]

        def evaluate(job):
            site, source, repo, units, pairs = job
            out = out_dir / ('%s-%s-%s.json' % (site, source, repo.name))
            result = self.process([binary, 'eval', '--site', site, '--repo', str(self.run.factory.root / 'repos' / repo.name),
                                   '--name', repo.name, '--units', str(units), '--pairs', str(pairs), '--json', str(out)],
                                  env=env, capture_output=True, text=True)
            if result.returncode:
                raise CampaignFault('selector_host_script_failed', details={'script': 'decide-rerank eval', 'site': site, 'repository': repo.name,
                                                                            'stderr': (result.stderr or result.stdout)[-2000:]})
            return json.loads(out.read_text())

        # Each evaluation runs its own engine, so a few run side by side.
        with concurrent.futures.ThreadPoolExecutor(max_workers=RERANK_WORKERS) as pool:
            return rerank_measure(list(pool.map(evaluate, jobs)))

    def preload(self):
        """The option vocabulary a host must serve the turn-load head, as the pinned
        checkout derives it from the catalog the head was trained on."""
        from ..run import model_dir
        return json.loads(self.host('preload.py', ['--corpus', self.catalog(), '--model', model_dir(self.run)]).stdout)

    def pilot(self, heads, values, preload, out):
        """The runner mount: engine, checkpoint, heads, and the checkout's config with the
        policy's thresholds and vocabulary."""
        from .. import build, fleet
        from ..run import model_dir
        build.pilot(self.run.path('build', 'runner-engine', 'bialy'), model_dir(self.run), heads, out)
        fleet.stage_config_overlay(Path(self.run.checkout) / 'lycaon/config', out, self.run.setting('engine_on', {}).get('deadline_ms'),
                                   self.run.collection_window, values)
        build.serve_vocabulary(Path(out) / 'overlay' / 'config', preload)
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


def unqualified(head, metric, value=None, **bound):
    return CampaignFault('selector_unqualified', details={'failures': [{'head': head, 'metric': metric, 'value': value, **bound}]})


def fit(run, split, directory, workbench, config, label='selector'):
    """Train the three heads on `split`'s train rows, calibrate on its validation rows,
    and freeze the policy under `directory` with the runner mount that serves it."""
    from .. import training
    directory = Path(directory)
    assignment = ancestry.load(run.path('ancestry.json'))
    missing = [name for name in ('train.jsonl', 'val.jsonl', 'development.jsonl') if not (Path(split) / name).exists()]
    if missing:
        raise CampaignFault('selector_split_missing', details={'files': missing})
    inputs = workbench.inputs(Path(split))
    heads = {}
    for head, recipe in config['heads'].items():
        result = workbench.train(recipe, inputs, directory / 'heads', label)
        if 'head' not in result:
            raise unqualified(head, 'trained', result.get('skipped'))
        heads[head] = result['head']
    launcher = workbench.launcher(heads, directory / 'launcher.sh')
    catalog = json.loads(workbench.catalog().read_text())
    decisions = workbench.decisions(config)
    values, tuned = thresholds(config, decisions, workbench.calibrate(launcher, Path(split) / 'val.jsonl'), catalog)
    preload = workbench.preload()
    policy = {'schema': SCHEMA, 'heads': {head: {'recipe': config['heads'][head], 'sha256': file_digest(path)}
                                          for head, path in sorted(heads.items())},
              'backbone': {'model': training.BACKBONE[0], 'revision': training.BACKBONE[1], 'sha256': tree_digest(workbench.backbone())},
              'catalog_sha256': file_digest(workbench.catalog()), 'preload': preload,
              'decision_contract': decisions['version'], 'thresholds': values, 'calibrated': tuned,
              'fallback_rate': config['fallback_rate'], 'assignment_sha256': assignment.sha256}
    policy['identity'] = digest(policy)
    atomic_json(directory / 'p0.json', policy)
    corpus = calibrated_catalog(catalog, values, directory / 'corpus.json')
    pilot = workbench.pilot(heads, values, preload, directory / 'pilot')
    return {'policy': policy, 'heads': heads, 'launcher': str(launcher), 'corpus': str(corpus), 'pilot': str(pilot)}


def frozen(directory, trained=None):
    """The policy `directory` holds, refused unless it is intact: its identity is the
    digest of its content, and each named head file has the digest it pins."""
    directory = Path(directory)
    policy = json.loads((directory / 'p0.json').read_text())
    heads = {head: directory / 'heads' / (head + '.safetensors') for head in policy['heads']}
    if (digest({k: v for k, v in policy.items() if k != 'identity'}) != policy['identity'] or (trained and trained['policy'] != policy)
            or any(not path.is_file() or file_digest(path) != policy['heads'][head]['sha256'] for head, path in heads.items())):
        raise CampaignFault('selector_policy_changed')
    return policy, heads


def qualify(run, split, directory, trained, workbench, config):
    """Measure a frozen policy on `split`'s development rows and record its qualification.
    Any probe or bar it misses raises selector_unqualified with every miss; nothing is
    recorded then, so an unqualified policy has no qualification to release under."""
    directory = Path(directory)
    policy, heads = frozen(directory, trained)
    rows = Path(split) / 'development.jsonl'
    parity = workbench.parity(trained['launcher'], trained['corpus'], heads['turn-load'], rows, **config['parity'])
    replay = workbench.replay(trained['launcher'], trained['corpus'], rows, directory / 'development.json')
    code_rank = workbench.rerank(heads['code-rank'], 'development', directory / 'rerank-development')
    atomic_json(directory / 'rerank-development.json', code_rank)
    missed = failures(config['bars'], {'turn-load': replay, 'unit-rank': replay, 'code-rank': code_rank})
    if not parity['passed']:
        missed.insert(0, {'head': 'turn-load', 'metric': 'parity', 'value': parity['worst_gap'], 'at_most': parity['tolerance'],
                          'summary': parity['summary']})
    if missed:
        raise CampaignFault('selector_unqualified', details={'failures': missed})
    measures = {head: {rule['metric']: measure(source, rule['metric']) for rule in config['bars'][head]}
                for head, source in (('turn-load', replay), ('unit-rank', replay), ('code-rank', code_rank))}
    record = {'schema': QUALIFICATION, 'policy': policy['identity'], 'qualified': True, 'parity': parity, 'bars': config['bars'],
              'measures': measures}
    atomic_json(directory / 'qualification.json', record)
    return record


def qualification(directory):
    """The recorded qualification of the policy `directory` holds, or None."""
    path = Path(directory) / 'qualification.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    policy, _ = frozen(directory)
    return record if record.get('qualified') is True and record.get('policy') == policy['identity'] else None


def fallback_measures(rows):
    """The engine-off fallback on the same rows: it offers every loadable tool, so it reaches every
    called one and its precision is the share of offered tools that were called."""
    offered = called = 0
    for row in rows:
        loadable = set(row['offered']['loadable'])
        offered += len(loadable)
        called += len(loadable & set(row['labels']['tools']))
    return {'recall': 1.0 if called else None, 'precision': round(called / offered, 3) if offered else None,
            'bytes_saved_per_turn': 0}


def holdout(run, split, directory, trained, workbench):
    """One held-out comparison of a qualified policy against the engine-off fallback: the
    turn decisions on the held-out rows, and code-rank on the held-out repositories
    against their text-match order. It is read once and decides nothing."""
    directory = Path(directory)
    path, rows = directory / 'holdout.json', Path(split) / 'holdout.jsonl'
    policy, heads = frozen(directory, trained)
    if path.exists() and json.loads(path.read_text())['policy'] == policy['identity']:
        return json.loads(path.read_text())
    if qualification(directory) is None:
        raise CampaignFault('selector_unqualified', details={'failures': [{'head': None, 'metric': 'qualification', 'value': None}]})
    report = workbench.replay(trained['launcher'], trained['corpus'], rows, directory / 'holdout-replay.json', truth='called')
    tools = report['overall']['tools']['_all']
    comparison = {'schema': 'pw-selector-holdout/2', 'policy': policy['identity'], 'truth': 'called',
                  'p0': {'recall': tools.get('recall'), 'precision': tools.get('precision'),
                         'bytes_saved_per_turn': report['overall'].get('savings', {}).get('bytes_saved_per_turn')},
                  'fallback': fallback_measures(json.loads(line) for line in rows.read_text().splitlines() if line.strip()),
                  'turns': report['overall'].get('turns'),
                  'code_rank': workbench.rerank(heads['code-rank'], 'system_test', directory / 'rerank-holdout')}
    atomic_json(path, comparison)
    return comparison


class SelectorTraining:
    """The campaign's fit: P0 from the head-row run's shared splits."""

    def __init__(self, journal, run, workbench=None, config=None):
        self.journal, self.run = journal, run
        self.workbench, self.config = workbench or Workbench(run), config or load()
        self.directory = journal.directory / 'selector'

    def execute(self, inputs, attempt):
        if ancestry.load(self.run.path('ancestry.json')).sha256 != inputs['rows']['assignment_sha256']:
            raise CampaignFault('selector_split_not_shared')
        return fit(self.run, self.run.path('split'), self.directory, self.workbench, self.config)

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        frozen(self.directory, output)


class SelectorQualification:
    """Qualify the three heads together on development; P0 is whole or absent."""

    def __init__(self, journal, run, workbench=None, config=None):
        self.journal, self.run = journal, run
        self.workbench, self.config = workbench or Workbench(run), config or load()
        self.directory = journal.directory / 'selector'

    def output(self, record, trained):
        policy = trained['policy']
        if record is None or record['policy'] != policy['identity']:
            raise CampaignFault('selector_qualification_changed')
        return {'qualified': True, 'policy': policy['identity'],
                'configuration': {'pilot': trained['pilot'], 'policy': policy['identity'],
                                  'thresholds': policy['thresholds'], 'fallback_rate': policy['fallback_rate']}}

    def execute(self, inputs, attempt):
        trained = inputs['selector']
        return self.output(qualify(self.run, self.run.path('split'), self.directory, trained, self.workbench, self.config), trained)

    def reconcile(self, inputs, attempt):
        if qualification(self.directory) is not None:
            return self.output(qualification(self.directory), inputs['selector'])
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output != self.output(qualification(self.directory), inputs['selector']):
            raise CampaignFault('selector_qualification_changed')


class SelectorHoldout:
    """The campaign's held-out comparison of P0, after it qualified."""

    def __init__(self, journal, run, workbench=None):
        self.journal, self.run = journal, run
        self.workbench = workbench or Workbench(run)
        self.directory = journal.directory / 'selector'

    def execute(self, inputs, attempt):
        return holdout(self.run, self.run.path('split'), self.directory, inputs['selector'], self.workbench)

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if json.loads((self.directory / 'holdout.json').read_text()) != output:
            raise CampaignFault('selector_holdout_changed')
