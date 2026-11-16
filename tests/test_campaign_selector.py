import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from selector_bench import CATALOG, DECISIONS, PARITY, PRELOAD, RERANK, campaign, qualified, rows

from bialy import build, coderank, fleet, host_scripts
from bialy import run as runmod
from bialy.campaign import ancestry, selector
from bialy.campaign.journal import CampaignFault


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


def test_training_freezes_p0_from_train_and_validation_only(tmp_path):
    journal, assignment, run, workbench = campaign(tmp_path)
    stage = selector.SelectorTraining(journal, run, workbench)
    output = stage.execute(rows(assignment), 'a')
    stage.validate(rows(assignment), output)
    policy = output['policy']
    assert [recipe for recipe, _, _ in workbench.trained] == ['B5', 'E4', 'code-rank']
    assert workbench.calibrated_on == ['val.jsonl'] and workbench.replayed == [] and workbench.reranked == []
    assert policy['schema'] == 'pw-selector-policy/2'
    assert policy['thresholds']['turn.tools.load_at'] == 0.55 and policy['thresholds']['lookup.read_at'] == 2.8
    assert policy['calibrated'] == ['request.load_at', 'request.max_loads', 'request.nearest_loads', 'turn.tools.load_at']
    assert policy['decision_contract'] == 3 and policy['assignment_sha256'] == assignment.sha256
    assert {key: policy['backbone'][key] for key in ('model', 'revision')} == {
        'model': 'convaiinnovations/laya-multilingual', 'revision': 'e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'}
    Path(output['heads']['unit-rank']).write_text('replaced')
    with pytest.raises(CampaignFault, match='selector_policy_changed'):
        stage.validate(rows(assignment), output)
    journal.close()


def test_p0_binds_the_vocabulary_its_heads_were_trained_with_and_the_runners_serve_it(tmp_path):
    journal, assignment, run, workbench = campaign(tmp_path)
    output = selector.SelectorTraining(journal, run, workbench).execute(rows(assignment), 'a')
    assert output['policy']['preload'] == PRELOAD
    served = json.loads((Path(output['pilot']) / 'served.json').read_text())
    assert served == {'heads': sorted(selector.HEADS), 'thresholds': output['policy']['thresholds'], 'preload': PRELOAD}
    other = campaign(tmp_path / 'other', preload={**PRELOAD, 'options': {**PRELOAD['options'], 'edit': 'Edit one file in place'}})
    changed = selector.SelectorTraining(other[0], other[2], other[3]).execute(rows(other[1]), 'a')
    assert changed['policy']['identity'] != output['policy']['identity']
    journal.close()
    other[0].close()


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
    assert not (journal.directory / 'selector' / 'p0.json').exists()
    journal.close()


def test_every_head_is_qualified_on_development_rows_and_repositories(tmp_path):
    journal, run, workbench, trained = qualified(tmp_path)
    stage = selector.SelectorQualification(journal, run, workbench)
    output = stage.execute({'selector': trained}, 'q')
    assert output['qualified'] is True and output['configuration']['pilot'] == trained['pilot']
    assert output['configuration']['fallback_rate'] == selector.load()['fallback_rate']
    assert workbench.seen_corpus['questions']['tools']['load_at'] == 0.55
    # Each head's bar is measured where nothing trained it or chose its epoch or thresholds.
    assert workbench.replayed == [('development.jsonl', 'consensus')] and workbench.reranked == [('code-rank.safetensors', 'development')]
    assert workbench.probed == [('turn-load.safetensors', 'development.jsonl', 24, 0.001)]
    record = selector.qualification(journal.directory / 'selector')
    assert record['policy'] == trained['policy']['identity'] and record['parity'] == PARITY
    assert record['measures']['code-rank'] == {'pairs': 240, 'mrr_gain': 0.11}
    assert stage.reconcile({'selector': trained}, 'q') == output and len(workbench.replayed) == 1
    stage.validate({'selector': trained}, output)
    journal.close()


def test_no_bar_reads_a_measure_the_trainer_wrote(tmp_path):
    for rules in selector.load()['bars'].values():
        assert not [rule for rule in rules if rule['metric'].startswith('val_')]
    journal, run, workbench, trained = qualified(tmp_path)
    Path(trained['heads']['code-rank']).with_suffix('.metrics.json').write_text(json.dumps({'val_acc': 0.99}))
    workbench.rerank_measure = {**RERANK, 'mrr_gain': -0.02}
    with pytest.raises(CampaignFault, match='selector_unqualified') as raised:
        selector.SelectorQualification(journal, run, workbench).execute({'selector': trained}, 'q')
    assert raised.value.details['failures'] == [{'head': 'code-rank', 'metric': 'mrr_gain', 'value': -0.02, 'at_least': 0.01}]
    journal.close()


@pytest.mark.parametrize('changes, head, metric', [
    ({'rerank': {'pairs': 0}}, 'code-rank', 'pairs'),
    ({'replay': {'overall': {'tools': {'_all': {'recall': 0.5, 'precision': 0.7}}, 'requests': {'wanted': {'top1': 0.6}}}}}, 'turn-load',
     'overall.tools._all.recall'),
    ({'parity': {**PARITY, 'passed': False, 'worst_gap': 0.2}}, 'turn-load', 'parity'),
    ({'parity': {'passed': False, 'tolerance': 0.001, 'summary': 'parity: unverified; the head records no encoding settings',
                 'worst_gap': None, 'turns': 0}}, 'turn-load', 'parity')])
def test_one_miss_fails_the_whole_policy_and_records_no_qualification(tmp_path, changes, head, metric):
    journal, run, workbench, trained = qualified(tmp_path, **changes)
    with pytest.raises(CampaignFault, match='selector_unqualified') as raised:
        selector.SelectorQualification(journal, run, workbench).execute({'selector': trained}, 'q')
    assert (raised.value.details['failures'][0]['head'], raised.value.details['failures'][0]['metric']) == (head, metric)
    assert selector.qualification(journal.directory / 'selector') is None
    with pytest.raises(CampaignFault, match='selector_unqualified'):
        selector.SelectorHoldout(journal, run, workbench).execute({'selector': trained}, 'h')
    journal.close()


def test_a_qualification_does_not_outlive_its_policy(tmp_path):
    journal, run, workbench, trained = qualified(tmp_path)
    selector.SelectorQualification(journal, run, workbench).execute({'selector': trained}, 'q')
    directory = journal.directory / 'selector'
    record = json.loads((directory / 'qualification.json').read_text())
    (directory / 'qualification.json').write_text(json.dumps(dict(record, policy='0' * 64)))
    assert selector.qualification(directory) is None
    (directory / 'qualification.json').write_text(json.dumps(record))
    (directory / 'heads' / 'turn-load.safetensors').write_text('retrained')
    with pytest.raises(CampaignFault, match='selector_policy_changed'):
        selector.qualification(directory)
    journal.close()


def test_rerank_reports_pool_over_their_pairs():
    reports = [{'pairs': 30, 'lexical': {'mrr': 0.4, 'hit_1': 0.2}, 'blended': {'mrr': 0.6, 'hit_1': 0.4}, 'improved': 9, 'regressed': 1},
               {'pairs': 10, 'lexical': {'mrr': 0.8, 'hit_1': 0.6}, 'blended': {'mrr': 0.8, 'hit_1': 0.6}, 'improved': 0, 'regressed': 0}]
    assert selector.rerank_measure(reports) == {'pairs': 40, 'lexical': {'mrr': 0.5, 'hit_1': 0.3}, 'blended': {'mrr': 0.65, 'hit_1': 0.45},
                                                'improved': 9, 'regressed': 1, 'mrr_gain': 0.15}
    assert selector.rerank_measure([]) == {'pairs': 0}


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


def test_the_holdout_comparison_is_read_once_after_qualification(tmp_path):
    journal, run, workbench, trained = qualified(tmp_path)
    run.path('split', 'holdout.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in (
        {'offered': {'loadable': ['read', 'edit', 'command']}, 'labels': {'tools': ['read']}},
        {'offered': {'loadable': ['read', 'edit']}, 'labels': {'tools': ['edit', 'web']}})))
    stage = selector.SelectorHoldout(journal, run, workbench)
    with pytest.raises(CampaignFault, match='selector_unqualified'):
        stage.execute({'selector': trained}, 'early')
    assert workbench.replayed == []
    selector.SelectorQualification(journal, run, workbench).execute({'selector': trained}, 'q')
    output = stage.execute({'selector': trained}, 'h')
    assert output['p0'] == {'recall': 0.9, 'precision': 0.7, 'bytes_saved_per_turn': None}
    assert output['fallback'] == {'recall': 1.0, 'precision': 0.4, 'bytes_saved_per_turn': 0}
    assert output['code_rank'] == RERANK and workbench.reranked[-1] == ('code-rank.safetensors', 'system_test')
    assert workbench.replayed[-1] == ('holdout.jsonl', 'called')
    assert stage.execute({'selector': trained}, 'again') == output and len(workbench.replayed) == 2
    stage.validate({'selector': trained}, output)
    journal.close()


class Host:
    """A process double for the pinned checkout's scripts and binaries: each call is
    recorded and answered by `answer`."""

    def __init__(self, answer):
        self.calls, self.answer = [], answer

    def __call__(self, args, **kw):
        self.calls.append((list(args), kw))
        return subprocess.CompletedProcess(args, *self.answer(list(args)))


def real_bench(factory, tmp_path, monkeypatch, answer):
    run = runmod.Run(factory=dataclasses.replace(factory, root=tmp_path), name='t',
                     settings=runmod.settings_for(factory, {'lycaon_checkout': str(tmp_path / 'co')}))
    run.path().mkdir(parents=True)
    run.path('corpus.json').write_text(json.dumps(CATALOG))
    monkeypatch.setattr(runmod, 'model_dir', lambda run: tmp_path / 'model')
    host = Host(answer)
    return run, selector.Workbench(run, host), host


def test_the_parity_probe_runs_the_hosts_script_in_the_trainers_environment(factory, tmp_path, monkeypatch):
    from bialy import training
    monkeypatch.setattr(training, 'venv_python', lambda root, settings: '/venv/python')
    passing = (0, 'truth []\nparity: worst gap 0.0004 over 24 turns\n', '')
    run, bench, host = real_bench(factory, tmp_path, monkeypatch, lambda args: passing)
    report = bench.parity('/l.sh', '/corpus.json', '/heads/turn-load.safetensors', Path('/split/development.jsonl'), 24, 0.001)
    assert report == {'passed': True, 'tolerance': 0.001, 'summary': 'parity: worst gap 0.0004 over 24 turns', 'worst_gap': 0.0004, 'turns': 24}
    args, kw = host.calls[0]
    assert args[:2] == ['/venv/python', 'scripts/bialy/parity_probe.py'] and kw['cwd'] == str(tmp_path / 'co')
    assert args[args.index('--model') + 1] == 'convaiinnovations/laya-multilingual' and args[args.index('--n') + 1] == '24'
    assert args[args.index('--engine') + 1] == '/l.sh' and args[args.index('--examples') + 1] == '/split/development.jsonl'
    assert kw['env']['HF_HUB_OFFLINE'] == '1'
    for code, line in ((1, 'parity: worst gap 0.2100 over 24 turns'), (1, 'parity: the head was trained at max_len 512, head budget 192; the engine encodes at 1024, 512'),
                       (2, 'parity: unverified; no engine answers were compared')):
        _, failing, _ = real_bench(factory, tmp_path / str(len(line)), monkeypatch, lambda args, code=code, line=line: (code, line + '\n', ''))
        report = failing.parity('/l.sh', '/c', '/h', Path('/d.jsonl'), 24, 0.001)
        assert report['passed'] is False and report['summary'] == line


def test_every_host_script_is_called_with_flags_the_host_accepts(factory, tmp_path, monkeypatch):
    from bialy import training
    monkeypatch.setattr(training, 'venv_python', lambda root, settings: '/venv/python')
    recorded = json.loads((Path(__file__).parent / 'fixtures' / 'factory-contract.json').read_text())['scripts']
    answers = {'calibrate.py': 'turn:\n  tools:\n    load_at: 0.55\n', 'preload.py': json.dumps(PRELOAD), 'parity_probe.py': '', 'replay_eval.py': ''}

    def answer(args):
        script = args[1][len('scripts/bialy/'):]
        if '--json' in args:
            Path(args[args.index('--json') + 1]).write_text(json.dumps({'overall': {}}))
        return (0, answers[script], '')
    run, bench, host = real_bench(factory, tmp_path, monkeypatch, answer)
    assert bench.calibrate('/l.sh', Path('/split/val.jsonl')) == {'turn': {'tools': {'load_at': 0.55}}}
    assert bench.replay('/l.sh', '/corpus.json', Path('/split/development.jsonl'), tmp_path / 'replay.json', truth='called') == {'overall': {}}
    assert bench.preload() == PRELOAD
    bench.parity('/l.sh', '/corpus.json', '/h', Path('/split/development.jsonl'), 24, 0.001)
    assert sorted(args[1] for args, _ in host.calls) == ['scripts/bialy/' + name for name in sorted(answers)]
    for args, _ in host.calls:
        script = args[1][len('scripts/bialy/'):]
        assert {a for a in args if a.startswith('--')} <= set(recorded[script]) and {a for a in args if a.startswith('--')} <= set(host_scripts.FLAGS[script])
    preload = next(args for args, _ in host.calls if args[1].endswith('preload.py'))
    assert preload[preload.index('--corpus') + 1] == str(run.path('corpus.json')) and preload[preload.index('--model') + 1] == str(tmp_path / 'model')
    _, failing, _ = real_bench(factory, tmp_path / 'f', monkeypatch, lambda args: (2, '', 'usage: replay_eval.py: error: unrecognized arguments'))
    with pytest.raises(CampaignFault, match='selector_host_script_failed'):
        failing.calibrate('/l.sh', Path('/split/val.jsonl'))


def test_the_launcher_and_the_runner_mount_come_from_the_training_corpus_and_policy(factory, tmp_path, monkeypatch):
    run, bench, _ = real_bench(factory, tmp_path, monkeypatch, lambda args: (0, '', ''))
    text = bench.launcher({'turn-load': '/h/turn-load.safetensors'}, tmp_path / 'l.sh').read_text()
    assert '--head-max-len %d ' % CATALOG['state']['head_tokens'] in text and str(run.path('build', 'host-engine', 'bialy')) in text
    config = tmp_path / 'co' / 'lycaon/config'
    (config / build.RELEASE).parent.mkdir(parents=True)
    (config / build.RELEASE).write_text(json.dumps({'release': 'shipped', 'preload': {'catalog_revision': 'old', 'options': {'read': 'Read'}}}))
    (config / build.RELEASE).with_name('decisions.yaml').write_text(yaml.safe_dump(DECISIONS))
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model' / '.complete').write_text('')
    run.path('build', 'runner-engine').mkdir(parents=True)
    run.path('build', 'runner-engine', 'bialy').write_bytes(b'engine')
    head = tmp_path / 'turn-load.safetensors'
    head.write_bytes(b'h')
    mount = bench.pilot({'turn-load': head}, {'turn.tools.load_at': 0.55}, PRELOAD, tmp_path / 'pilot')
    served = json.loads((mount / 'overlay/config' / build.RELEASE).read_text())
    assert served['preload'] == PRELOAD and served['release'] == 'shipped'
    assert yaml.safe_load((mount / fleet.DECISIONS).read_text())['turn']['tools']['load_at'] == 0.55
    assert (mount / 'heads' / 'turn-load.safetensors').read_bytes() == b'h' and (mount / 'bialy').read_bytes() == b'engine'
    assert fleet.pilot_env(mount)['LYCAON_CONFIG_ROOT'] == fleet.PILOT_MOUNT + '/overlay'


def test_code_rank_is_measured_through_the_engine_on_one_partitions_repositories(factory, tmp_path, monkeypatch):
    def answer(args):
        Path(args[args.index('--json') + 1]).write_text(json.dumps(
            {'pairs': 10, 'lexical': {'mrr': 0.4, 'hit_1': 0.2}, 'blended': {'mrr': 0.5, 'hit_1': 0.3}, 'improved': 2, 'regressed': 1}))
        return (0, '', '')
    run, bench, host = real_bench(factory, tmp_path, monkeypatch, answer)
    repos = runmod.scaled_factory(run).repos[:3]
    graph = ancestry.Graph()
    for repo, partition in zip(repos, ('train', 'development', 'system_test'), strict=True):
        graph.add('repository:' + repo.name, 'repository', source_class='synthetic', holdout=partition)
        run.path('coderank', 'units').mkdir(parents=True, exist_ok=True)
        run.path('coderank', 'pairs').mkdir(parents=True, exist_ok=True)
        run.path('coderank', 'units', repo.name + '.jsonl').write_text('{}\n')
        run.path('coderank', 'pairs', 'docs-%s.jsonl' % repo.name).write_text('{}\n')
        run.path('coderank', 'pairs', 'model-%s.jsonl' % repo.name).write_text('')
    ancestry.allocate(graph, 7, {'train': 1.0}).save(run.path('ancestry.json'))
    measure = bench.rerank('/heads/code-rank.safetensors', 'development', tmp_path / 'rerank')
    assert measure['pairs'] == 10 * len(coderank.SITES) and measure['mrr_gain'] == 0.1
    assert {args[args.index('--name') + 1] for args, _ in host.calls} == {repos[1].name}
    args, kw = host.calls[0]
    assert args[1] == 'eval' and '--no-engine' not in args and '--dump' not in args
    assert kw['env']['LYCAON_DECIDE_HEADS'] == 'code-rank=/heads/code-rank.safetensors'
    assert kw['env']['LYCAON_DECIDE_BINARY'] == str(run.path('build', 'host-engine', 'bialy'))
    assert bench.rerank('/heads/code-rank.safetensors', 'validation', tmp_path / 'none') == {'pairs': 0}
