import json
from pathlib import Path

import pytest

from bialy.provenance import file_sha256
from bialy.student_host import exporting
from bialy.student_host.jobs import JobError
from bialy.student_host.resources import write
from bialy.student_host.specification import load


@pytest.mark.parametrize('probe', ['passed', 'failed', {'status': 'failed'}])
def test_export_requires_bound_artifacts_and_explicit_probe_success(tmp_path, monkeypatch, probe):
    write(tmp_path / 'plan.json', {'checkout': str(tmp_path / 'host'), 'models': str(tmp_path / 'models'),
          'specification': load(), 'student': 'e4b', 'out': str(tmp_path / 'training'),
          'specification_sha256': 'spec', 'host_commit': 'host'})
    monkeypatch.setattr(exporting, 'validate_inputs', lambda plan: None)
    monkeypatch.setattr(exporting, 'validate_result', lambda plan: None)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        directory = Path(args[args.index('--out') + 1])
        directory.mkdir()
        if directory.name == 'merged':
            write(directory / 'manifest.json', {'files': 'fixture'})
        else:
            for name in ('model-f16.gguf', 'model-Q4_K_M.gguf'):
                (directory / name).write_bytes(name.encode())
            write(directory / 'manifest.json', {'schema': 'pw-custom-gguf/1', 'load_probe': probe,
                  'llama_cpp_revision': 'revision', 'merged_manifest_sha256': file_sha256(tmp_path / 'merged/manifest.json'),
                  'unquantized': {'sha256': file_sha256(directory / 'model-f16.gguf')},
                  'quantized': {'sha256': file_sha256(directory / 'model-Q4_K_M.gguf')}})
    if probe == 'passed':
        result = exporting.export(tmp_path, tmp_path / 'llama', 'revision', run=run)
        assert result['context'] == load()['context']
        assert json.loads((tmp_path / 'artifact.json').read_text()) == result
    else:
        with pytest.raises(JobError, match='parity receipt'):
            exporting.export(tmp_path, tmp_path / 'llama', 'revision', run=run)
        assert not (tmp_path / 'artifact.json').exists()
    assert len(calls) == 2


def test_only_a_returned_remote_artifact_skips_local_export(tmp_path):
    assert not exporting.returned(tmp_path)
    write(tmp_path / 'artifact.json', {'unquantized_retained': True})
    assert not exporting.returned(tmp_path)
    write(tmp_path / 'artifact.json', {'unquantized_retained': False})
    assert exporting.returned(tmp_path)
