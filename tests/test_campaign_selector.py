import json
from pathlib import Path

import pytest
import yaml

from bialy import fleet
from bialy.campaign import ancestry, selector
from bialy.campaign.journal import CampaignFault, Journal

DECISIONS = {'version': 3,
             'turn': {'tools': {'load_at': 0.6}, 'skills': {'preload_at': 2.8, 'margin': 0.3}},
             'request': {'load_at': 1.8, 'max_loads': 4, 'nearest_loads': 0},
             'lookup': {'read_at': 2.8, 'margin': 0.3},
             'tool_event': {'read_at': 2.8, 'pointer_at': 2.4, 'margin': 0.3}}
CATALOG = {'questions': {'tools': {'load_at': 0.6}, 'skills': {'preload_at': 2.8, 'margin': 0.3}},
           'request': {'load_at': 1.8, 'max_loads': 4, 'nearest_loads': 0}, 'lookup': {'read_at': 2.8, 'margin': 0.3}}
CALIBRATED = {'turn': {'tools': {'load_at': 0.55}}, 'request': {'load_at': 2.1, 'max_loads': 3, 'nearest_loads': 1}}
REPLAY = {'overall': {'tools': {'_all': {'recall': 0.9, 'precision': 0.7}}, 'requests': {'wanted': {'top1': 0.6}}}}
METRICS = {'val_acc': 0.7, 'val_loss': 0.4}
REPOSITORIES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa']


class Run:
    def __init__(self, root):
        self.root = root

    def path(self, *parts):
        return self.root.joinpath(*parts)


class Workbench:
    def __init__(self, root, replay=REPLAY, metrics=METRICS, calibrated=CALIBRATED, skip=None):
        self.root, self.replay_report, self.metrics, self.calibrated, self.skip = root, replay, metrics, calibrated, skip
        self.replays, self.trained = 0, []
        root.mkdir(parents=True, exist_ok=True)
        (root / 'corpus.json').write_text(json.dumps(CATALOG))
        (root / 'backbone').mkdir(exist_ok=True)
        (root / 'backbone' / 'model.safetensors').write_text('weights')

    def inputs(self, split, dumps):
        return {'train': split / 'train.jsonl', 'val': split / 'val.jsonl', 'dumps_train': dumps['train'], 'dumps_val': dumps['validation']}

    def train(self, recipe, inputs, heads_dir):
        self.trained.append((recipe, inputs))
        if recipe == self.skip:
            return {'skipped': 'no rows'}
        heads_dir.mkdir(parents=True, exist_ok=True)
        head = heads_dir / ({'B5': 'turn-load', 'E4': 'unit-rank'}.get(recipe, recipe) + '.safetensors')
        head.write_text(recipe)
        if recipe == 'code-rank':
            head.with_suffix('.metrics.json').write_text(json.dumps(self.metrics))
        return {'head': str(head)}

    def launcher(self, heads, path):
        path.write_text(json.dumps(heads))
        return path

    def calibrate(self, launcher, rows):
        assert rows.name == 'val.jsonl'
        return self.calibrated

    def replay(self, launcher, corpus, rows, out, truth='consensus'):
        assert rows.name in ('development.jsonl', 'holdout.jsonl')
        self.truth = truth
        self.replays += 1
        self.seen_corpus = json.loads(Path(corpus).read_text())
        Path(out).write_text(json.dumps(self.replay_report))
        return self.replay_report

    def decisions(self, config):
        return DECISIONS

    def catalog(self):
        return self.root / 'corpus.json'

    def backbone(self):
        return self.root / 'backbone'

    def pilot(self, heads_dir, values, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / 'thresholds.json').write_text(json.dumps(values))
        return out


def rows(assignment):
    return {'rows': {'assignment_sha256': assignment.sha256}}


def campaign(tmp_path, **changes):
    graph = ancestry.Graph()
    for repository in REPOSITORIES:
        graph.add('repository:' + repository, 'task', source_class='synthetic')
    assignment = ancestry.allocate(graph, 7, {'train': 0.4, 'validation': 0.3, 'development': 0.3},
                                   {'train': 2, 'validation': 2, 'development': 2})
    run = Run(tmp_path / 'run')
    assignment.save(run.path('ancestry.json'))
    for name in ('train.jsonl', 'val.jsonl', 'development.jsonl'):
        run.path('split').mkdir(parents=True, exist_ok=True)
        run.path('split', name).write_text('{}\n')
    run.path('coderank', 'dumps').mkdir(parents=True)
    for repository in REPOSITORIES + ['outside']:
        run.path('coderank', 'dumps', 'callers-scan-%s.jsonl' % repository).write_text('{}\n')
    journal = Journal(tmp_path / 'campaign', {})
    workbench = Workbench(tmp_path / 'bench', **changes)
    return journal, assignment, run, workbench


def test_the_configuration_names_three_heads_their_bars_and_a_fallback_rate():
    config = selector.load()
    assert set(config['heads']) == set(selector.HEADS) == set(config['bars'])
    assert 0 < config['fallback_rate'] < 1


def test_a_configuration_without_every_head_is_refused(tmp_path):
    config = selector.load()
    del config['heads']['code-rank']
    path = tmp_path / 'selector.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(CampaignFault, match='selector_configuration_invalid'):
        selector.load(path)


def test_calibration_replaces_only_what_it_printed_and_the_catalog_must_match_the_contract():
    config = selector.load()
    values, tuned = selector.thresholds(config, DECISIONS, {'turn': {'tools': {'load_at': 0.55}}}, CATALOG)
    assert values['turn.tools.load_at'] == 0.55 and tuned == ['turn.tools.load_at']
    assert values['turn.skills.preload_at'] == 2.8 and values['tool_event.pointer_at'] == 2.4
    assert values['request.max_loads'] == 4
    drifted = json.loads(json.dumps(CATALOG))
    drifted['lookup']['margin'] = 0.5
    with pytest.raises(CampaignFault, match='selector_catalog_contract_mismatch'):
        selector.thresholds(config, DECISIONS, {}, drifted)
    incomplete = json.loads(json.dumps(DECISIONS))
    del incomplete['tool_event']['margin']
    with pytest.raises(CampaignFault, match='selector_thresholds_missing'):
        selector.thresholds(config, incomplete, {}, CATALOG)


def test_the_replay_catalog_carries_the_policys_thresholds(tmp_path):
    values = {'turn.tools.load_at': 0.55, 'request.max_loads': 3, 'tool_event.margin': 0.3}
    written = json.loads(selector.calibrated_catalog(CATALOG, values, tmp_path / 'corpus.json').read_text())
    assert written['questions']['tools']['load_at'] == 0.55 and written['request']['max_loads'] == 3
    assert 'tool_event' not in written and CATALOG['questions']['tools']['load_at'] == 0.6


def test_dumps_follow_their_repositorys_partition_and_outside_dumps_are_unused(tmp_path):
    _, assignment, run, _ = campaign(tmp_path)
    dumps = selector.partitioned_dumps(run.path('coderank', 'dumps'), assignment)
    assert sum(len(items) for items in dumps.values()) == len(REPOSITORIES)
    for partition, items in dumps.items():
        assert all(assignment.partition('repository:' + item.stem.split('-', 2)[-1]) == partition for item in items)


def test_training_freezes_p0_from_train_and_validation_only(tmp_path):
    journal, assignment, run, workbench = campaign(tmp_path)
    stage = selector.SelectorTraining(journal, run, workbench)
    output = stage.execute(rows(assignment), 'a')
    stage.validate(rows(assignment), output)
    policy = output['policy']
    assert [recipe for recipe, _ in workbench.trained] == ['B5', 'E4', 'code-rank']
    _, inputs = workbench.trained[0]
    assert all(assignment.partition('repository:' + dump.stem.split('-', 2)[-1]) == 'train' for dump in inputs['dumps_train'])
    assert all(assignment.partition('repository:' + dump.stem.split('-', 2)[-1]) == 'validation' for dump in inputs['dumps_val'])
    assert policy['thresholds']['turn.tools.load_at'] == 0.55 and policy['thresholds']['lookup.read_at'] == 2.8
    assert policy['calibrated'] == ['request.load_at', 'request.max_loads', 'request.nearest_loads', 'turn.tools.load_at']
    assert policy['decision_contract'] == 3 and policy['assignment_sha256'] == assignment.sha256
    assert json.loads((Path(output['pilot']) / 'thresholds.json').read_text()) == policy['thresholds']
    Path(output['heads']['unit-rank']).write_text('replaced')
    with pytest.raises(CampaignFault, match='selector_policy_changed'):
        stage.validate(rows(assignment), output)
    journal.close()


def test_p0_identity_follows_its_thresholds(tmp_path):
    first = campaign(tmp_path / 'a')
    second = campaign(tmp_path / 'b', calibrated={'turn': {'tools': {'load_at': 0.5}}})
    identities = [selector.SelectorTraining(journal, run, bench).execute(rows(assignment), 'a')['policy']['identity']
                  for journal, assignment, run, bench in (first, second)]
    assert identities[0] != identities[1]
    first[0].close()
    second[0].close()


def test_training_refuses_a_split_from_another_assignment(tmp_path):
    journal, _, run, workbench = campaign(tmp_path)
    with pytest.raises(CampaignFault, match='selector_split_not_shared'):
        selector.SelectorTraining(journal, run, workbench).execute({'rows': {'assignment_sha256': '0' * 64}}, 'a')
    journal.close()


def test_a_head_that_cannot_train_leaves_the_selector_unqualified(tmp_path):
    journal, assignment, run, workbench = campaign(tmp_path, skip='E4')
    with pytest.raises(CampaignFault, match='selector_unqualified') as raised:
        selector.SelectorTraining(journal, run, workbench).execute(rows(assignment), 'a')
    assert raised.value.details['failures'][0]['head'] == 'unit-rank'
    journal.close()


def test_qualification_passes_only_when_every_head_clears_its_bar(tmp_path):
    journal, assignment, run, workbench = campaign(tmp_path)
    trained = selector.SelectorTraining(journal, run, workbench).execute(rows(assignment), 'a')
    stage = selector.SelectorQualification(journal, run, workbench)
    output = stage.execute({'selector': trained}, 'q')
    assert output['qualified'] is True and output['configuration']['pilot'] == trained['pilot']
    assert output['configuration']['fallback_rate'] == selector.load()['fallback_rate']
    assert workbench.seen_corpus['questions']['tools']['load_at'] == 0.55
    assert stage.reconcile({'selector': trained}, 'q') == output and workbench.replays == 1
    stage.validate({'selector': trained}, output)
    journal.close()


def test_one_head_short_of_its_bar_fails_the_whole_policy(tmp_path):
    journal, assignment, run, workbench = campaign(tmp_path, metrics={'val_acc': 0.3})
    trained = selector.SelectorTraining(journal, run, workbench).execute(rows(assignment), 'a')
    with pytest.raises(CampaignFault, match='selector_unqualified') as raised:
        selector.SelectorQualification(journal, run, workbench).execute({'selector': trained}, 'q')
    assert raised.value.details['failures'] == [{'head': 'code-rank', 'metric': 'val_acc', 'value': 0.3, 'at_least': 0.6}]
    journal.close()


def test_a_missing_measure_is_a_failure_not_a_pass():
    bars = {'turn-load': [{'metric': 'overall.tools._all.recall', 'at_least': 0.85}]}
    assert selector.failures(bars, {'turn-load': {'overall': {}}})[0]['value'] is None


def test_planned_fallbacks_are_drawn_deterministically_at_the_policys_rate():
    p0 = {'pilot': '/pilot', 'policy': 'abc', 'fallback_rate': 0.1}
    draws = [selector.stratum(p0, 'task-%d' % n, n % 3) for n in range(2000)]
    assert draws == [selector.stratum(p0, 'task-%d' % n, n % 3) for n in range(2000)]
    assert set(draws) == {'p0', 'planned_fallback'} and 0.07 < draws.count('planned_fallback') / len(draws) < 0.13
    assert selector.stratum({'qualified': True, 'mode': 'explicit_tool_loading'}, 'task-1', 0) == 'conservative'
    assert selector.plan_record(p0) == {'policy': 'abc', 'fallback_rate': 0.1}


def test_the_pilot_overlay_carries_thresholds_and_becomes_the_hosts_config(tmp_path):
    config = tmp_path / 'config'
    decisions = config / 'packs/painted-wolf/platform/host/decisions.yaml'
    decisions.parent.mkdir(parents=True)
    decisions.write_text(yaml.safe_dump(DECISIONS))
    pilot = tmp_path / 'pilot'
    (pilot / 'heads').mkdir(parents=True)
    (pilot / 'heads' / 'turn-load.safetensors').write_text('h')
    assert 'LYCAON_CONFIG_ROOT' not in fleet.pilot_env(pilot)
    fleet.stage_config_overlay(config, pilot, thresholds={'turn.tools.load_at': 0.55, 'request.max_loads': 3})
    written = yaml.safe_load((pilot / fleet.DECISIONS).read_text())
    assert written['turn']['tools']['load_at'] == 0.55 and written['request']['max_loads'] == 3
    assert fleet.pilot_env(pilot)['LYCAON_CONFIG_ROOT'] == fleet.PILOT_MOUNT + '/overlay'
    with pytest.raises(ValueError, match='no threshold'):
        fleet.stage_config_overlay(config, pilot, thresholds={'turn.tools.unload_at': 0.1})


def test_the_holdout_comparison_is_read_once_against_the_fallback(tmp_path):
    journal, assignment, run, workbench = campaign(tmp_path)
    trained = selector.SelectorTraining(journal, run, workbench).execute(rows(assignment), 'a')
    run.path('split', 'holdout.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in (
        {'offered': {'loadable': ['read', 'edit', 'command']}, 'labels': {'tools': ['read']}},
        {'offered': {'loadable': ['read', 'edit']}, 'labels': {'tools': ['edit', 'web']}})))
    stage = selector.SelectorHoldout(journal, run, workbench)
    output = stage.execute({'selector': trained}, 'h')
    assert output['p0'] == {'recall': 0.9, 'precision': 0.7, 'bytes_saved_per_turn': None} and workbench.truth == 'called'
    assert output['fallback'] == {'recall': 1.0, 'precision': 0.4, 'bytes_saved_per_turn': 0}
    assert stage.execute({'selector': trained}, 'again') == output and workbench.replays == 1
    stage.validate({'selector': trained}, output)
    journal.close()
