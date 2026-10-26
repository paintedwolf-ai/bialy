import json

import pytest

from bialy.provenance import file_sha256
from bialy.student_host.jobs import JobError
from bialy.student_host.transfer import checkpoints


def test_partial_transfer_does_not_replace_verified_generation(tmp_path):
    incoming, durable = tmp_path / 'incoming', tmp_path / 'durable'
    first = incoming / '00000001-step0001'
    first.mkdir(parents=True)
    (first / 'state.pt').write_bytes(b'checkpoint')
    (first / 'manifest.json').write_text(json.dumps({'identity': 'run', 'state_sha256': file_sha256(first / 'state.pt')}))
    assert checkpoints(incoming, durable, 'run') == [first.name]
    (first / 'state.pt').write_bytes(b'interrupted transfer')
    assert checkpoints(incoming, durable, 'run') == []
    assert (durable / first.name / 'state.pt').read_bytes() == b'checkpoint'
    with pytest.raises(JobError, match='foreign checkpoint'):
        checkpoints(incoming, durable, 'another-run')
