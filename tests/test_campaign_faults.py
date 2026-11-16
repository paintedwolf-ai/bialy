import json
from pathlib import Path

import pytest
import yaml
from campaign_world import Disconnect, Stage, World, outputs
from host_bundles import Image, contract, write_episode

from bialy import host_bundle
from bialy.campaign import runtime
from bialy.campaign.execution import Executor, ReceiptBackend
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.sources import exported_episode, verified_exports
from bialy.campaign.student import training_fault
from bialy.provenance import file_sha256
from bialy.student_host.jobs import JobError
from bialy.student_host.transfer import checkpoints

STUDENTS = ['e4b', '12b', '26b-a4b']


def planned(tmp_path):
    value = {'profile': 'pilot', 'epoch': 'campaign', 'checkout': '/host', 'host_commit': 'a' * 40,
             'host_image': 'bialy-runner:campaign-off', 'factory_root': '/factory', 'corpus': '/corpus.json', 'models': '/models',
             'teachers': ['teacher'], 'reviewers': ['first', 'second'], 'server_image': 'sha256:' + 'c' * 64, 'gateway': '192.0.2.1',
             'llama': '/llama', 'llama_revision': 'b' * 40, 'full_training': 'local', 'device': 'cpu', 'mirror': '/mirror'}
    path = tmp_path / 'campaign.yaml'
    path.write_text(yaml.safe_dump(value))
    runtime.plan(path, tmp_path / 'plan')
    return tmp_path / 'plan'


def composer(world):
    values = outputs(STUDENTS)
    return lambda journal, directory, body, adopted: {name: Stage(world, name, output) for name, output in values.items()}


@pytest.mark.parametrize('stage,ledger', [('collection', 'provider_requests'), ('train_12b', 'rentals')])
def test_a_controller_disconnect_resumes_without_a_second_request_or_rental(tmp_path, stage, ledger):
    plan, world = planned(tmp_path), World()
    world.crash = {stage}
    with pytest.raises(Disconnect):
        runtime.drive(plan, composer=composer(world))
    assert runtime.status(plan / 'run')['stages'][stage] == 'running'
    result = runtime.drive(plan, composer=composer(world))
    assert result['status'] == 'qualified'
    assert len(world.owned(stage, getattr(world, ledger))) == 1
    assert len(world.provider_requests) == 1 and len(world.rentals) == len(STUDENTS)


def test_a_disconnect_before_any_paid_work_is_not_guessed_complete(tmp_path):
    plan, world = planned(tmp_path), World()
    journal = Journal(plan / 'run', {'schema': 'pw-campaign-run/1', 'plan_sha256': runtime.read_plan(plan)['plan_sha256']})
    journal.begin('collection', {'host': outputs(STUDENTS)['host_qualification'], 'teachers': outputs(STUDENTS)['teacher_qualification'],
                                 'selector': outputs(STUDENTS)['host_qualification']['conservative_reference']})
    journal.close()
    for name in ('host_qualification', 'reviewer_qualification', 'teacher_qualification', 'selector_supplied'):
        world.provider_requests.append((name, 'unrelated'))
    result = runtime.drive(plan, composer=composer(world))
    assert result['status'] == 'insufficient_qualified_data'
    assert runtime.status(plan / 'run')['stages']['collection'] == 'failed'
    assert world.owned('collection', world.provider_requests) == []


def test_running_out_of_memory_fails_once_and_is_never_retried(tmp_path):
    class Training:
        calls = 0

        def execute(self, inputs, attempt):
            Training.calls += 1
            job = tmp_path / 'job'
            out = tmp_path / 'out'
            job.mkdir(exist_ok=True)
            out.mkdir(exist_ok=True)
            (job / 'plan.json').write_text(json.dumps({'out': str(out)}))
            (out / 'terminal.json').write_text(json.dumps({'failure': {'code': 'out_of_memory', 'retryable': False}}))
            raise training_fault(job, 1)

        reconcile = execute

        def validate(self, inputs, output):
            raise AssertionError('no output')
    journal = Journal(tmp_path / 'run', {})
    backend = ReceiptBackend(tmp_path / 'receipts', {'train_26b-a4b': Training()})
    for _ in range(2):
        result = Executor(journal, backend, lambda _: None).run('train_26b-a4b', {})
    assert result['fault']['code'] == 'student_out_of_memory' and result['tries'] == 1 and Training.calls == 1
    journal.close()


def test_a_missing_or_changed_bundle_is_refused_before_any_row_is_admitted(tmp_path):
    bundle = write_episode(tmp_path / 'bundle')
    inputs = {str(bundle): file_sha256(bundle / 'manifest.json')}
    assert exported_episode(inputs, inputs[str(bundle)]).root_session == 'root'
    (bundle / 'operations.jsonl').write_text('{"id": "forged"}\n')
    with pytest.raises(CampaignFault, match='host_capture_changed'):
        exported_episode(inputs, inputs[str(bundle)])
    (bundle / 'effects.jsonl').unlink()
    with pytest.raises(host_bundle.HostEvidenceError, match='host_bundle_changed'):
        host_bundle.Episode(bundle)
    (bundle / 'manifest.json').write_text('{not json')
    with pytest.raises(host_bundle.HostEvidenceError, match='host_bundle_missing'):
        host_bundle.Episode(bundle)
    (bundle / 'manifest.json').unlink()
    with pytest.raises(CampaignFault, match='host_capture_changed'):
        exported_episode(inputs, inputs[str(bundle)])
    with pytest.raises(CampaignFault, match='row_source_not_in_host_export'):
        exported_episode(inputs, 'f' * 64)


def test_an_export_whose_bundle_digest_changed_is_refused(tmp_path):
    bundle = write_episode(tmp_path / 'bundle')
    rows = tmp_path / 'export' / 'rows.jsonl'
    rows.parent.mkdir()
    rows.write_text('')
    manifest = rows.parent / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'pw-training-rows/1', 'identity': {'host_commit': 'host'},
                                    'bundles': {str(bundle): 'e' * 64}, 'rows_sha256': file_sha256(rows)}))
    with pytest.raises(CampaignFault, match='host_export_input_changed'):
        verified_exports([manifest], 'host')


@pytest.mark.parametrize('changes,code', [({'schema': 'pw-training-contract/2'}, 'host_contract_unsupported'),
                                          ({'host_modified': True}, 'host_contract_revision_mismatch')])
def test_a_contract_of_another_version_or_a_modified_host_is_refused(tmp_path, changes, code):
    class Changed(Image):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            path = next(Path(spec.split(':')[0]) for flag, spec in zip(command, command[1:], strict=False) if flag == '-v')
            target = path / 'contract' / 'contract.json'
            target.write_text(json.dumps({**contract(), **changes}))
            return result
    with pytest.raises(host_bundle.HostEvidenceError, match=code):
        host_bundle.contract('runner:off', tmp_path / 'contract', 'c' * 40, Changed())


def generation(root, name, identity, state, recorded=None):
    path = root / name
    path.mkdir(parents=True)
    (path / 'state.pt').write_bytes(state)
    (path / 'manifest.json').write_text(json.dumps({'identity': identity,
                                                     'state_sha256': recorded or file_sha256(path / 'state.pt')}))


def test_a_corrupt_newest_checkpoint_leaves_the_latest_verified_one_to_resume_from(tmp_path):
    incoming, durable = tmp_path / 'incoming', tmp_path / 'durable'
    generation(incoming, '1', 'job', b'one')
    generation(incoming, '2', 'job', b'two', recorded='0' * 64)
    assert checkpoints(incoming, durable, 'job') == ['1']
    assert sorted(path.name for path in durable.iterdir()) == ['1']


def test_a_mirror_that_disagrees_with_the_replica_is_refused_and_the_replica_kept(tmp_path):
    incoming, durable = tmp_path / 'incoming', tmp_path / 'durable'
    generation(incoming, '1', 'job', b'one')
    checkpoints(incoming, durable, 'job')
    kept = (durable / '1' / 'state.pt').read_bytes()
    replaced = tmp_path / 'replaced'
    generation(replaced, '1', 'job', b'other')
    with pytest.raises(JobError, match='replica collision'):
        checkpoints(replaced, durable, 'job')
    assert (durable / '1' / 'state.pt').read_bytes() == kept
