from pathlib import Path
from types import SimpleNamespace

import pytest

from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.student import Training
from bialy.provenance import file_sha256
from bialy.student_host.specification import DEFAULT, identity, load


def setup(tmp_path):
    journal = Journal(tmp_path / 'campaign', {})
    splits = tmp_path / 'splits'
    splits.mkdir()
    for name in ('train.jsonl', 'val.jsonl', 'holdout.jsonl', 'source.json'):
        (splits / name).write_text('{}\n')
    configuration = {'specification': str(DEFAULT), 'full_training': 'local', 'checkout': '/host',
                     'models': '/models', 'mirror': '/mirror', 'device': 'cuda:0', 'host_commit': 'a' * 40}
    inputs = {'dataset': {'students': {'e4b': {'directory': str(splits),
              'files': {p.name: file_sha256(p) for p in splits.iterdir()},
              'specification_sha256': identity(load())}}}}
    return journal, configuration, inputs


def test_training_uses_authenticated_split_and_private_key_environment(tmp_path, monkeypatch):
    journal, configuration, inputs = setup(tmp_path)
    observed = []
    def create(root, spec, student, checkout, models, splits, mirror, device, pilot, *, dry_run=False):
        assert student == 'e4b' and pilot is True and spec['context']['thinking'] is False
        assert dry_run is False
        job = root / 'unique'
        job.mkdir(parents=True)
        (job / 'plan.json').write_text('{}')
        return job
    def run(arguments, **kwargs):
        assert arguments[-2] == 'train'
        key = Path(kwargs['env']['PW_TRAINING_SOURCE_KEY'])
        assert key.stat().st_mode & 0o077 == 0
        observed.append(arguments)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('bialy.campaign.student.jobs.create', create)
    handler = Training(journal, configuration, 'e4b', pilot=True, run=run)
    monkeypatch.setattr(handler, 'result', lambda job: {'job': str(job)})
    result = handler.execute(inputs, 'attempt')
    assert handler.reconcile(inputs, 'attempt') == result
    assert len(observed) == 2 and observed[0][-1] == observed[1][-1]
    assert len(list((handler.directory('attempt') / 'jobs').iterdir())) == 1
    journal.close()


def test_changed_dataset_is_rejected_before_training(tmp_path):
    journal, configuration, inputs = setup(tmp_path)
    Path(inputs['dataset']['students']['e4b']['directory'], 'train.jsonl').write_text('changed')
    handler = Training(journal, configuration, 'e4b')
    with pytest.raises(CampaignFault, match='dataset_changed'):
        handler.execute(inputs, 'attempt')
    journal.close()


def test_unsettled_unknown_job_is_not_recreated(tmp_path):
    journal, configuration, inputs = setup(tmp_path)
    handler = Training(journal, configuration, 'e4b')
    with pytest.raises(CampaignFault, match='binding_unknown'):
        handler.reconcile(inputs, 'attempt')
    assert not handler.directory('attempt').exists()
    journal.close()


def test_training_failure_cannot_be_reported_as_completion(tmp_path):
    journal, configuration, inputs = setup(tmp_path)
    handler = Training(journal, configuration, 'e4b', run=lambda *args, **kwargs: SimpleNamespace(returncode=127))
    job = tmp_path / 'job'
    job.mkdir()
    with pytest.raises(CampaignFault, match='training_failed') as caught:
        handler.train_local(job)
    assert caught.value.details['exit_status'] == 127
    journal.close()


@pytest.mark.parametrize('pilot', [False, True])
def test_pilot_and_full_training_use_the_configured_machine(tmp_path, monkeypatch, pilot):
    journal, configuration, inputs = setup(tmp_path)
    configuration.update(full_training='remote', remote='machine.json')
    calls = []
    def train(config, student, splits, key, out, **kwargs):
        calls.append((config, student, kwargs))
        directory = kwargs.get('resume') or out / 'instance'
        if not kwargs.get('resume'):
            directory.mkdir(parents=True)
        return {'directory': str(directory)}
    monkeypatch.setattr('bialy.campaign.student.remote_training.train', train)
    handler = Training(journal, configuration, 'e4b', pilot=pilot)
    monkeypatch.setattr(handler, 'result', lambda job: {'job': str(job)})
    result = handler.execute(inputs, 'attempt')
    assert handler.reconcile(inputs, 'attempt') == result
    assert len(calls) == 2 and all(call[2]['pilot'] is pilot for call in calls)
    assert calls[0][2].get('resume') is None
    assert calls[1][2]['resume'].name == 'instance'
    journal.close()
