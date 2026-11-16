import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bialy.campaign.baseline import Baseline
from bialy.campaign.journal import CampaignFault
from bialy.provenance import file_sha256

TEMPLATE = 't' * 64
RENDERING = {'profile': 'gemma4-code-v1', 'template_sha256': TEMPLATE}


class Host:
    """Stands in for base_checkpoint.py and gguf_export.py, writing what each one writes."""

    def __init__(self, files, staged_template=TEMPLATE, served_template=TEMPLATE, schema='pw-custom-gguf/2'):
        self.files, self.staged_template, self.served_template, self.schema = files, staged_template, served_template, schema
        self.calls = []

    def __call__(self, args, **kwargs):
        script = Path(next(arg for arg in args if arg.endswith('.py'))).name
        self.calls.append((script, args))
        out = Path(args[args.index('--out') + 1])
        out.mkdir(exist_ok=True)
        if script == 'base_checkpoint.py':
            (out / 'manifest.json').write_text(json.dumps({'schema': 'pw-base-checkpoint/2', 'source': self.files,
                'rendering_profile': 'gemma4-code-v1', 'template_sha256': self.staged_template}))
            return
        staged = Path(args[args.index('--merge') + 1])
        (out / 'model-Q4_K_M.gguf').write_bytes(b'base')
        served = {'rendering': {'template_sha256': self.served_template, 'cases': ['declared tools and parallel calls']}, 'behavior': None}
        (out / 'manifest.json').write_text(json.dumps({'schema': self.schema, 'llama_cpp_revision': 'pinned',
            'merged_manifest_sha256': file_sha256(staged / 'manifest.json'), 'template_sha256': self.staged_template,
            'behavior_sha256': None, 'served': {'unquantized': served, 'quantized': served}}))


def baseline(tmp_path, monkeypatch, **host):
    files = {'config.json': 'c' * 64}
    monkeypatch.setattr('bialy.campaign.baseline.checkpoint', lambda *args: files)
    monkeypatch.setattr('bialy.campaign.baseline.rendering', lambda specification, checkout: RENDERING)
    cfg = {'models': str(tmp_path / 'models'), 'checkout': str(tmp_path / 'host'), 'host_commit': 'host',
           'llama': str(tmp_path / 'llama'), 'llama_revision': 'pinned'}
    run = Host(files, **host)
    return Baseline(SimpleNamespace(directory=tmp_path / 'run'), cfg, {'models': {'e4b': {'directory': 'base'}}}, 'e4b', run=run), run


def test_the_base_arm_is_the_stock_weights_in_the_training_template(tmp_path, monkeypatch):
    stage, run = baseline(tmp_path, monkeypatch)
    output = stage.execute({}, 'attempt')
    stage.validate({}, output)
    assert output['template_sha256'] == TEMPLATE
    assert [script for script, _ in run.calls] == ['base_checkpoint.py', 'gguf_export.py']
    staging, export = (args for _, args in run.calls)
    assert staging[staging.index('--model') + 1] == str(tmp_path / 'models/base')
    assert export[export.index('--merge') + 1] == staging[staging.index('--out') + 1]
    assert export[export.index('--server') + 1].endswith('build/bin/llama-server') and '--resume' not in export
    stage.reconcile({}, 'attempt')
    assert [script for script, _ in run.calls][2:] == ['gguf_export.py'] and '--resume' in run.calls[2][1]
    (Path(output['directory']) / 'model-Q4_K_M.gguf').write_bytes(b'changed')
    with pytest.raises(CampaignFault, match='baseline_artifact_changed'):
        stage.validate({}, output)


@pytest.mark.parametrize('host', [{'staged_template': 's' * 64}, {'served_template': 's' * 64}, {'schema': 'pw-custom-gguf/1'}])
def test_a_base_export_in_another_template_or_never_served_is_refused(tmp_path, monkeypatch, host):
    stage, _ = baseline(tmp_path, monkeypatch, **host)
    with pytest.raises(CampaignFault, match='baseline_export_unqualified'):
        stage.execute({}, 'attempt')


def test_a_changed_checkpoint_or_template_pin_invalidates_the_base_arm(tmp_path, monkeypatch):
    stage, _ = baseline(tmp_path, monkeypatch)
    output = stage.execute({}, 'attempt')
    monkeypatch.setattr('bialy.campaign.baseline.rendering', lambda specification, checkout: {**RENDERING, 'template_sha256': 'u' * 64})
    with pytest.raises(CampaignFault, match='baseline_artifact_changed'):
        stage.validate({}, output)
    with pytest.raises(CampaignFault, match='baseline_inputs_changed'):
        stage.execute({}, 'attempt')
