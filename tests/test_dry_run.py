import json
from types import SimpleNamespace

import pytest

from bialy.campaign import dry_run
from bialy.campaign.dry_run_data import import_evidence, qualified, read_source
from bialy.campaign.episodes import Episodes
from bialy.campaign.journal import CampaignFault, Journal, canonical
from bialy.campaign.lineage import Lineage
from bialy.campaign.pilot_bank import build
from bialy.student_host.select_rows import select
from bialy.student_host.specification import arguments, load


def test_select_keeps_whole_rows_and_reports_exclusions():
    rows = [{'tokens': list(range(count)), 'target': {'text': str(count)}} for count in (12, 4, 7, 5)]
    chosen, report = select(rows, lambda row: {'input_ids': row['tokens']}, 8, 2)
    assert chosen == [rows[1], rows[3]]
    assert chosen[0] is rows[1]
    assert report == {'source_rows': 4, 'selected_rows': 2, 'oversize_rows': 1,
                      'selected_tokens': [4, 5], 'source_indices': [1, 3]}
    with pytest.raises(ValueError, match='insufficient complete'):
        select(rows, lambda row: {'input_ids': row['tokens']}, 3, 1)


def test_dry_run_pauses_after_one_update_and_resumes_without_that_bound():
    spec = load()
    args = arguments(spec, 'e4b', '/models', '/splits', '/out', 'cuda:0', '/mirror', dry_run=True)
    assert args[args.index('--checkpoint-every') + 1] == '1'
    assert args[args.index('--max-updates') + 1] == '1'
    assert args[args.index('--max-length') + 1] == '65536'
    resumed = arguments(spec, 'e4b', '/models', '/splits', '/out', 'cuda:0', '/mirror',
                        dry_run=True, resume=True)
    assert '--resume' in resumed and '--max-updates' not in resumed
    ordinary = arguments(spec, 'e4b', '/models', '/splits', '/out', 'cuda:0', '/mirror', pilot=True)
    assert ordinary[ordinary.index('--max-updates') + 1] == '10'


def prepare_evidence(journal, bank):
    lineage = Lineage(journal)
    lineage.roots(build(bank))
    lineage.freeze()
    Episodes(journal, lineage)
    journal.db.execute('CREATE TABLE judge_controls(identity TEXT PRIMARY KEY, receipt TEXT NOT NULL)')
    journal.db.commit()
    return lineage


def test_source_evidence_is_copied_without_rewriting_parent_or_terminal_failure(tmp_path):
    parent = Journal(tmp_path / 'parent', {})
    lineage = prepare_evidence(parent, tmp_path / 'bank')
    lineage.bind_session('epoch', 'session', 'train-structured_closeout')
    stage = parent.begin('collection_pilot', {})
    parent.settle('collection_pilot', stage['attempt'], 'passed', output={'qualified': True})
    stage = parent.begin('pilot_e4b', {})
    parent.settle('pilot_e4b', stage['attempt'], 'failed', fault={'code': 'out_of_memory', 'retryable': False})
    parent.close()
    original = (tmp_path / 'parent/campaign.sqlite').read_bytes()
    child = Journal(tmp_path / 'child', {})
    prepare_evidence(child, tmp_path / 'child-bank')
    copied = import_evidence(child, tmp_path / 'parent')
    assert copied.session_task('epoch', 'session')['id'] == 'train-structured_closeout'
    assert (tmp_path / 'parent/campaign.sqlite').read_bytes() == original
    with read_source(tmp_path / 'parent') as source:
        assert qualified(source, 'collection_pilot') == {'qualified': True}
        with pytest.raises(CampaignFault, match='source_stage_unqualified'):
            qualified(source, 'pilot_e4b')
    child.db.execute("UPDATE tasks SET spec='changed' WHERE id='train-structured_closeout'")
    child.db.commit()
    with pytest.raises(CampaignFault, match='source_evidence_changed'):
        import_evidence(child, tmp_path / 'parent')
    child.close()


class Handler:
    def __init__(self, output):
        self.output = output

    def execute(self, inputs, attempt):
        return self.output

    reconcile = execute

    def validate(self, inputs, output):
        assert output == self.output


def test_dry_run_drives_suffix_without_claiming_quality(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    cfg = {'checkout': '/host', 'models': '/models', 'host_commit': 'host'}
    parent = Journal(source, {'configuration': cfg})
    outputs = {'collection_pilot': {}, 'host_qualification': {'conservative_reference': {}},
               'teacher_baseline': {}, 'baseline_e4b': {'directory': '/base'},
               'evaluate_base_e4b': {'attempts': [{'task_id': 'system_test-structured_closeout', 'success': False}]}}
    for name, output in outputs.items():
        stage = parent.begin(name, {})
        parent.settle(name, stage['attempt'], 'passed', output=output)
    parent.close()
    attempt = {'stage_status': 'passed', 'effective_thinking': False, 'success': False}
    handlers = {'export_e4b': Handler({'directory': '/artifact'}),
                'evaluate_trained_e4b': Handler({'attempts': [attempt]})}
    plan = {'tasks': [{'id': 'system_test-structured_closeout'}], 'lineage_sha256': 'frozen'}
    monkeypatch.setattr(dry_run, 'runtime_configuration', lambda cfg, directory, host: cfg)
    monkeypatch.setattr(dry_run, 'compose', lambda *args: (handlers, plan))
    monkeypatch.setattr(dry_run, 'import_evidence', lambda *args: SimpleNamespace())
    monkeypatch.setattr(dry_run, 'Data', lambda *args: Handler({'students': {}}))
    monkeypatch.setattr(dry_run, 'Training', lambda *args, **kwargs: Handler({'job': '/job'}))
    result = dry_run.run(source, tmp_path / 'dry')
    assert result['status'] == 'mechanical_pipeline_complete'
    assert result['promotion_qualified'] is False
    assert result['evaluation']['attempts'][0]['success'] is False
    assert result['limits']['training_rows'] == 2
    assert json.loads((tmp_path / 'dry/result.json').read_text()) == result
    assert dry_run.run(source, tmp_path / 'dry')['status'] == result['status']
    assert canonical(result['plan'])

def test_dry_run_has_a_new_epoch_and_factory_namespace(tmp_path, monkeypatch):
    source = {"Id": "sha256:pinned", "Config": {"Labels": {"pw.host_commit": "host"}}}
    calls = []
    def process(args, **kwargs):
        calls.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0 if args[-1] == "sha256:pinned" else 1,
                                   stdout=json.dumps([source]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(dry_run, "prepare_fleet", lambda _: None)
    cfg = {"epoch": "source-epoch", "host_image": "bialy-runner:source-epoch-off", "host_commit": "host"}
    result = dry_run.runtime_configuration(cfg, tmp_path, {"image": "sha256:pinned"}, process)
    assert result["epoch"] != cfg["epoch"]
    assert result["factory_root"] == str(tmp_path / "factory")
    assert calls[-1] == ["docker", "tag", "sha256:pinned", result["host_image"]]
    assert cfg["epoch"] == "source-epoch"
