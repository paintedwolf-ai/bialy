import json
from pathlib import Path

import pytest

from bialy.student_host.jobs import JobError, checkpoint
from bialy.student_host.materialize import models
from bialy.student_host.specification import load


def test_downloads_every_pinned_student_and_reuses_verified_files(tmp_path):
    calls = []
    def download(args, **kwargs):
        calls.append(args)
        target = Path(args[args.index('--local-dir') + 1])
        (target / 'weights.safetensors').write_bytes(b'fixture')
        (target / '.cache').mkdir()
        (target / '.cache/token').write_text('download cache')
    spec = load()
    result = models(spec, tmp_path, run=download)
    assert set(result) == set(spec['models'])
    assert len(calls) == len(spec['models'])
    assert {args[args.index('--revision') + 1] for args in calls} == {m['revision'] for m in spec['models'].values()}
    assert models(spec, tmp_path, run=download) == result
    assert len(calls) == len(spec['models'])
    for name in result:
        assert '.cache/token' not in checkpoint(spec, name, tmp_path)
    target = Path(result['e4b']['directory']) / 'weights.safetensors'
    target.write_bytes(b'changed')
    with pytest.raises(JobError, match='content changed'):
        models(spec, tmp_path, run=download)


def test_a_wrong_model_revision_cannot_start_training(tmp_path):
    spec = load()
    def download(args, **kwargs):
        (Path(args[-1]) / 'weights').write_bytes(b'fixture')
    models(spec, tmp_path, run=download)
    receipt = tmp_path / (spec['models']['e4b']['directory'] + '.source.json')
    value = json.loads(receipt.read_text())
    value['source']['revision'] = '0' * 40
    receipt.write_text(json.dumps(value))
    with pytest.raises(JobError, match='source or content'):
        checkpoint(spec, 'e4b', tmp_path)


def test_a_job_downloads_only_its_selected_student(tmp_path):
    spec = load()
    calls = []
    def download(args, **kwargs):
        calls.append(args[2])
        (Path(args[args.index('--local-dir') + 1]) / 'weights').write_bytes(b'fixture')
    assert set(models(spec, tmp_path, run=download, students=['26b-a4b'])) == {'26b-a4b'}
    assert calls == [spec['models']['26b-a4b']['repository']]
    for selection in ([], ['26b-a4b', '26b-a4b'], ['undeclared']):
        with pytest.raises(JobError, match='declared students'):
            models(spec, tmp_path, run=download, students=selection)
