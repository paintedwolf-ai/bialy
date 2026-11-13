import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest

from bialy.provenance import file_sha256
from bialy.student_host import exporting, remote_training
from bialy.student_host.jobs import JobError, tree
from bialy.student_host.resources import write
from bialy.student_host.specification import load

TEMPLATE = '{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}\n'


def safetensors(path, count):
    """A real safetensors file with `count` one-float tensors."""
    header = {'t%d' % n: {'dtype': 'F32', 'shape': [1], 'data_offsets': [4 * n, 4 * n + 4]} for n in range(count)}
    header['__metadata__'] = {}
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(4 * count))


class Worker:
    """Stands in for export.py, gguf_export.py and convert_mlx.py, writing what each one writes."""

    def __init__(self, probe='passed', template=TEMPLATE, mlx_template=TEMPLATE, interrupt_mlx=False, shards=False):
        self.probe, self.template, self.mlx_template = probe, template, mlx_template
        self.interrupt_mlx, self.shards, self.calls = interrupt_mlx, shards, []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        script = Path(next(arg for arg in args if arg.endswith('.py'))).name
        getattr(self, script.removesuffix('.py'))(args)

    def export(self, args):
        out = Path(args[args.index('--out') + 1])
        model = out / 'model'
        if self.shards:
            safetensors(model / 'model-00001-of-00002.safetensors', 2)
            safetensors(model / 'model-00002-of-00002.safetensors', 2)
            write(model / exporting.INDEX, {'metadata': {}, 'weight_map': {
                't0': 'model-00001-of-00002.safetensors', 't1': 'model-00002-of-00002.safetensors'}})
        else:
            safetensors(model / 'model.safetensors', 3)
        (model / 'chat_template.jinja').write_text(self.template)
        write(out / 'manifest.json', {'schema': 'pw-custom-merge/1', 'files': tree(model)})

    def gguf_export(self, args):
        out = Path(args[args.index('--out') + 1])
        out.mkdir(exist_ok=True)
        for name in (exporting.UNQUANTIZED, exporting.QUANTIZED):
            (out / name).write_bytes(name.encode())
        merge = Path(args[args.index('--merge') + 1])
        quantized = {'sha256': file_sha256(out / exporting.QUANTIZED), 'tensors': {'a': {}, 'b': {}},
                     'metadata': {'tokenizer.chat_template': self.template}}
        write(out / 'manifest.json', {'schema': 'pw-custom-gguf/1', 'load_probe': self.probe, 'llama_cpp_revision': 'revision',
                                      'merged_manifest_sha256': file_sha256(merge / 'manifest.json'),
                                      'unquantized': {'sha256': file_sha256(out / exporting.UNQUANTIZED)}, 'quantized': quantized})

    def convert_mlx(self, args):
        out = Path(args[args.index('--output') + 1])
        assert not out.exists()
        safetensors(out / 'model.safetensors', 5)
        if self.interrupt_mlx:
            self.interrupt_mlx = False
            raise KeyboardInterrupt
        (out / 'chat_template.jinja').write_text(self.mlx_template)
        write(out / 'conversion.json', {'schema': 'paintedwolf-mlx-conversion/2', 'profile': '26b-a4b',
                                        'source_sha256': args[args.index('--source-sha256') + 1],
                                        'recipe': 'affine-group64-f16', 'weights_sha256': file_sha256(out / 'model.safetensors'),
                                        'chat_template_sha256': file_sha256(out / 'chat_template.jinja')})


def job(root, monkeypatch):
    """A trained job whose host checkout carries the training template."""
    host = root / 'host'
    (host / exporting.TEMPLATE).parent.mkdir(parents=True)
    (host / exporting.TEMPLATE).write_text(TEMPLATE)
    training = root / 'training'
    safetensors(training / 'adapter/adapter_model.safetensors', 4)
    write(training / 'adapter/adapter_config.json', {'r': 16})
    (training / 'adapter/tokenizer.json').write_text('{}')
    write(training / 'result.json', {'adapter': tree(training / 'adapter')})
    directory = root / 'job'
    directory.mkdir()
    write(directory / 'plan.json', {'checkout': str(host), 'models': str(root / 'models'), 'specification': load(),
                                    'student': '26b-a4b', 'out': str(training), 'specification_sha256': 'spec',
                                    'host_commit': 'host'})
    monkeypatch.setattr(exporting, 'validate_inputs', lambda plan: None)
    monkeypatch.setattr(exporting, 'validate_result', lambda plan: None)
    return directory


def test_one_merge_returns_three_outputs_bound_by_the_receipt(tmp_path, monkeypatch):
    directory = job(tmp_path, monkeypatch)
    worker = Worker()
    artifact = exporting.export(directory, tmp_path / 'llama', 'revision', run=worker)
    assert [Path(next(a for a in call if a.endswith('.py'))).name for call in worker.calls] == [
        'export.py', 'gguf_export.py', 'convert_mlx.py']
    receipt = json.loads((directory / 'export_receipt.json').read_text())
    assert artifact['export_receipt_sha256'] == file_sha256(directory / 'export_receipt.json')
    assert artifact['context'] == load()['context'] and json.loads((directory / 'artifact.json').read_text()) == artifact
    outputs = receipt['outputs']
    assert {name: output['tensors'] for name, output in outputs.items()} == {'vulkan': 2, 'mlx': 5, 'lora_adapter': 4}
    assert outputs['mlx']['source_sha256'] == file_sha256(directory / 'merged/model/model.safetensors')
    assert outputs['mlx']['profile'] == '26b-a4b' and outputs['mlx']['quantization'] == 'affine-group64-f16'
    assert exporting.UNQUANTIZED not in outputs['vulkan']['files']
    assert set(outputs['lora_adapter']['files']) == set(exporting.ADAPTER)
    template = file_sha256(tmp_path / 'host' / exporting.TEMPLATE)
    assert outputs['vulkan']['chat_template_sha256'] == outputs['mlx']['chat_template_sha256'] == template
    for output in outputs.values():
        assert output['sha256'] == file_sha256(directory / output['directory'] / output['payload'])
        assert output['bytes'] == (directory / output['directory'] / output['payload']).stat().st_size


@pytest.mark.parametrize('probe', ['failed', {'status': 'failed'}])
def test_a_failed_load_probe_publishes_nothing(tmp_path, monkeypatch, probe):
    directory = job(tmp_path, monkeypatch)
    with pytest.raises(JobError, match='parity receipt'):
        exporting.export(directory, tmp_path / 'llama', 'revision', run=Worker(probe=probe))
    assert not (directory / 'artifact.json').exists()


@pytest.mark.parametrize('worker', [Worker(template='stock template'), Worker(mlx_template='stock template')])
def test_an_output_without_the_training_template_is_refused(tmp_path, monkeypatch, worker):
    directory = job(tmp_path, monkeypatch)
    with pytest.raises(JobError, match='training chat template'):
        exporting.export(directory, tmp_path / 'llama', 'revision', run=worker)
    assert not (directory / 'artifact.json').exists()


def test_an_export_interrupted_after_the_gguf_resumes_into_all_three(tmp_path, monkeypatch):
    directory = job(tmp_path, monkeypatch)
    worker = Worker(interrupt_mlx=True)
    with pytest.raises(KeyboardInterrupt):
        exporting.export(directory, tmp_path / 'llama', 'revision', run=worker)
    assert (directory / 'mlx/model.safetensors').exists() and not (directory / 'mlx/conversion.json').exists()
    resumed = Worker()
    resumed.export = resumed.gguf_export = lambda args: None
    artifact = exporting.export(directory, tmp_path / 'llama', 'revision', run=resumed)
    assert all('--resume' in call for call in resumed.calls[:2])
    assert set(json.loads((directory / 'export_receipt.json').read_text())['outputs']) == {'vulkan', 'mlx', 'lora_adapter'}
    exporting.export(directory, tmp_path / 'llama', 'revision', run=resumed)
    assert len(resumed.calls) == 5
    assert json.loads((directory / 'artifact.json').read_text()) == artifact


def test_a_changed_output_or_adapter_breaks_the_binding(tmp_path, monkeypatch):
    directory = job(tmp_path, monkeypatch)
    exporting.export(directory, tmp_path / 'llama', 'revision', run=Worker())
    (directory / 'mlx/tokenizer.json').write_text('{}')
    with pytest.raises(JobError, match='receipt does not match'):
        exporting.publish(directory, 'revision', unquantized=True)
    (directory / 'mlx/tokenizer.json').unlink()
    exporting.publish(directory, 'revision', unquantized=True)
    (directory / 'lora_adapter/adapter_config.json').write_text('{"r": 8}')
    with pytest.raises(JobError, match='retained adapter changed'):
        exporting.retain_adapter(tmp_path / 'training', directory / 'lora_adapter')


def test_an_adapter_unlike_the_training_result_is_not_retained(tmp_path, monkeypatch):
    directory = job(tmp_path, monkeypatch)
    write(tmp_path / 'training/adapter/adapter_config.json', {'r': 32})
    with pytest.raises(JobError, match='does not match its result'):
        exporting.retain_adapter(tmp_path / 'training', directory / 'lora_adapter')
    assert not (directory / 'lora_adapter').exists()


def test_a_sharded_merge_pins_the_index_and_every_shard(tmp_path):
    Worker(shards=True).export(['export.py', '--out', str(tmp_path / 'merged')])
    files = json.loads((tmp_path / 'merged/manifest.json').read_text())['files']
    names = sorted([exporting.INDEX, 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'])
    lines = ''.join('%s  %s\n' % (name, files[name]) for name in names)
    assert exporting.mlx_source(tmp_path / 'merged') == hashlib.sha256(lines.encode()).hexdigest()
    index = tmp_path / 'merged/model' / exporting.INDEX
    write(index, {'weight_map': {}})
    with pytest.raises(JobError, match='index is not the merged one'):
        exporting.mlx_source(tmp_path / 'merged')
    manifest = json.loads((tmp_path / 'merged/manifest.json').read_text())
    del manifest['files']['model-00002-of-00002.safetensors']
    write(index, {'weight_map': {'t1': 'model-00002-of-00002.safetensors'}})
    manifest['files'][exporting.INDEX] = file_sha256(index)
    write(tmp_path / 'merged/manifest.json', manifest)
    with pytest.raises(JobError, match='shard is missing'):
        exporting.mlx_source(tmp_path / 'merged')


def test_a_conversion_of_another_checkpoint_is_refused(tmp_path, monkeypatch):
    directory = job(tmp_path, monkeypatch)
    worker = Worker()
    worker.convert_mlx = lambda args, convert=worker.convert_mlx: convert([*args[:-1], 'other'])
    with pytest.raises(JobError, match='another checkpoint'):
        exporting.export(directory, tmp_path / 'llama', 'revision', run=worker)


class TrainingMachine:
    """A finished remote export: receives copy from a worker directory as rsync would."""
    host = 'fixture'

    def __init__(self, worker_directory):
        self.worker, self.commands, self.received = worker_directory, [], []

    def command(self, args, **kwargs):
        self.commands.append(args)
        return '4242\n' if args[:2] == ['bash', '-lc'] else json.dumps({'exit': 0, 'alive': False})

    def receive(self, source, destination, exclude=(), timeout=600):
        self.received.append((source, tuple(exclude)))
        local = self.worker / source.removeprefix('/w/job/runs/one/')
        destination.mkdir(parents=True, exist_ok=True)
        if local.is_file():
            shutil.copy(local, destination / local.name)
        else:
            shutil.copytree(local, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('.pending-*', *exclude))


def test_a_remote_export_returns_all_three_outputs_and_leaves_the_merge(tmp_path, monkeypatch):
    (tmp_path / 'worker').mkdir()
    worker_directory = job(tmp_path / 'worker', monkeypatch)
    exporting.export(worker_directory, tmp_path / 'llama', 'revision', run=Worker())
    destination = tmp_path / 'instance'
    (destination / 'job').mkdir(parents=True)
    shutil.copy(worker_directory / 'plan.json', destination / 'job/plan.json')
    machine = TrainingMachine(worker_directory)
    config = {'llama_commit': 'revision', 'machine': {'uv': '/opt/uv/bin/uv'}}
    artifact = remote_training.export_remotely(machine, config, '/w/job', '/w/job/runs/one', destination,
                                               lambda: None, lambda _: None)
    received = destination / 'job'
    assert artifact['unquantized_retained'] is False
    assert not (received / 'gguf' / exporting.UNQUANTIZED).exists() and not (received / 'merged/model').exists()
    assert {name for name in exporting.OUTPUTS if (received / name).is_dir()} == set(exporting.OUTPUTS)
    assert (received / 'export_receipt.json').read_bytes() == (worker_directory / 'export_receipt.json').read_bytes()
    assert ('/w/job/runs/one/mlx/', ()) in machine.received


def test_only_a_returned_remote_artifact_skips_local_export(tmp_path):
    assert not exporting.returned(tmp_path)
    write(tmp_path / 'artifact.json', {'unquantized_retained': True})
    assert not exporting.returned(tmp_path)
    write(tmp_path / 'artifact.json', {'unquantized_retained': False})
    assert exporting.returned(tmp_path)
