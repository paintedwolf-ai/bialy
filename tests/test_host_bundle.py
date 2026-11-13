import json

import pytest
from host_bundles import Image, contract, operation, write_episode

from bialy import host_bundle


def test_the_contract_is_exported_once_and_must_name_the_pinned_commit(tmp_path):
    image = Image()
    value, digest = host_bundle.contract('runner:off', tmp_path / 'contract', 'c' * 40, image)
    assert value['bundle'] == host_bundle.BUNDLE and len(digest) == 64
    host_bundle.contract('runner:off', tmp_path / 'contract', 'c' * 40, image)
    assert len(image.calls) == 1
    command = image.calls[0]
    assert command[command.index('--entrypoint') + 1] == host_bundle.ENTRYPOINT and '--network' in command and '--user' in command
    with pytest.raises(host_bundle.HostEvidenceError, match='host_contract_revision_mismatch'):
        host_bundle.contract('runner:off', tmp_path / 'contract', 'd' * 40, image)


def test_a_contract_from_an_unsupported_or_modified_host_is_refused(tmp_path):
    destination = tmp_path / 'contract'
    destination.mkdir()
    for changes, code in (({'bundle': 'pw-training-episode-bundle/9'}, 'host_contract_unsupported'),
                          ({'host_modified': True}, 'host_contract_revision_mismatch')):
        (destination / 'contract.json').write_text(json.dumps(contract(**changes)))
        with pytest.raises(host_bundle.HostEvidenceError, match=code):
            host_bundle.contract('runner:off', destination, 'c' * 40, Image())


def test_a_failed_export_keeps_its_diagnostic(tmp_path):
    with pytest.raises(host_bundle.HostEvidenceError, match='host_evidence_export_failed') as failure:
        host_bundle.export('runner:off', tmp_path / 'shard', 'root', tmp_path / 'bundle', process=Image(fail='training-bundle'))
    diagnostic = json.loads((tmp_path / 'bundle-failure.json').read_text())
    assert diagnostic == {'exit_status': 3, 'stderr': 'refused'} and failure.value.details['exit_status'] == 3
    with pytest.raises(host_bundle.HostEvidenceError, match='host_contract_export_failed'):
        host_bundle.contract('runner:off', tmp_path / 'contract', 'c' * 40, Image(fail='training-contract'))


def test_an_export_mounts_the_capture_read_only_and_reads_the_episode_back(tmp_path):
    image = Image(operations=[operation('r1', call='c1'), operation('e1', parent='r1')])
    episode = host_bundle.export('runner:off', tmp_path / 'shard', 'root', tmp_path / 'bundle', tmp_path / 'project', image)
    command = image.calls[0]
    assert str((tmp_path / 'shard').resolve()) + ':/capture:ro' in command and '--project' in command
    assert episode.root_session == 'root' and [(op['id'], [c['id'] for c in children]) for op, children in episode.calls()] == [('r1', ['e1'])]
    assert host_bundle.export('runner:off', tmp_path / 'shard', 'root', tmp_path / 'bundle', process=image).manifest_sha256 == episode.manifest_sha256
    assert len(image.calls) == 1
    with pytest.raises(host_bundle.HostEvidenceError, match='episode_host_identity_mismatch'):
        host_bundle.export('runner:off', tmp_path / 'shard', 'other', tmp_path / 'bundle', process=image)


def test_an_episode_refuses_changed_files_unlisted_or_changed_blobs_and_foreign_bundles(tmp_path):
    root = write_episode(tmp_path / 'bundle', sessions=[{'id': 'root', 'messages': [{'role': 'user', 'content': 'hi'}]}],
                         captures=[{'schema': 'pw-training-turn/1', 'request': {'reasoning_effort': 'none'}}],
                         observations=[{'kind': 'execution', 'value': {'status': 'complete'}}])
    episode = host_bundle.Episode(root)
    assert episode.messages(episode.sessions[0]) == [{'role': 'user', 'content': 'hi'}]
    assert episode.capture_request(episode.captures[0]) == {'reasoning_effort': 'none'}
    assert episode.capture_request({}) is None and episode.execution == {'status': 'complete'}
    with pytest.raises(host_bundle.HostEvidenceError, match='host_bundle_blob_unlisted'):
        episode.blob('f' * 64)
    (root / 'blobs' / episode.sessions[0]['messages_blob']).write_text('[]')
    with pytest.raises(host_bundle.HostEvidenceError, match='host_bundle_changed'):
        episode.messages(episode.sessions[0])
    (root / 'effects.jsonl').write_text('{}\n')
    with pytest.raises(host_bundle.HostEvidenceError, match='host_bundle_changed'):
        host_bundle.Episode(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    (root / 'manifest.json').write_text(json.dumps(dict(manifest, schema='other/1')))
    with pytest.raises(host_bundle.HostEvidenceError, match='host_bundle_unsupported'):
        host_bundle.Episode(root)
