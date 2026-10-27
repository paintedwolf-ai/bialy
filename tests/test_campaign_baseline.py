import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bialy.campaign.baseline import Baseline
from bialy.campaign.journal import CampaignFault
from bialy.provenance import file_sha256


def test_base_export_binds_unchanged_model_and_detects_artifact_changes(tmp_path, monkeypatch):
    model = tmp_path / 'models' / 'base'
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    files = {'config.json': file_sha256(model / 'config.json')}
    monkeypatch.setattr('bialy.campaign.baseline.checkpoint', lambda *args: files)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        merged = Path(args[args.index('--merge') + 1])
        out = Path(args[args.index('--out') + 1])
        out.mkdir(exist_ok=True)
        (out / 'model-Q4_K_M.gguf').write_bytes(b'base')
        (out / 'manifest.json').write_text(json.dumps({'load_probe': 'passed', 'llama_cpp_revision': 'pinned',
            'merged_manifest_sha256': file_sha256(merged / 'manifest.json')}))
    cfg = {'models': str(model.parent), 'checkout': str(tmp_path / 'host'), 'host_commit': 'host',
           'llama': str(tmp_path / 'llama'), 'llama_revision': 'pinned'}
    baseline = Baseline(SimpleNamespace(directory=tmp_path / 'run'), cfg,
                        {'models': {'e4b': {'directory': 'base'}}}, 'e4b', run=run)
    output = baseline.execute({}, 'attempt')
    baseline.validate({}, output)
    assert (tmp_path / 'run/baselines/e4b/attempt/checkpoint/model/config.json').read_text() == '{}'
    assert '--resume' not in calls[0]
    baseline.reconcile({}, 'attempt')
    assert '--resume' in calls[1]
    (Path(output['directory']) / 'model-Q4_K_M.gguf').write_bytes(b'changed')
    with pytest.raises(CampaignFault, match='baseline_artifact_changed'):
        baseline.validate({}, output)
