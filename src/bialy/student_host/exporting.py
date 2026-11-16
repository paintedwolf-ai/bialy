"""Train-to-export transitions with checked host parity receipts.

An export has three outputs: the Vulkan GGUF in `gguf/` and the MLX checkpoint in `mlx/`,
both from one fp32 merge, and the trained LoRA adapter in `lora_adapter/`.
`export_receipt.json` binds all three to
the template the rows were rendered with and to the merge's behaviour reference, which
the GGUF was served and held to before the receipt is written.
"""
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

from .. import host_contract
from ..provenance import file_sha256
from .jobs import PIPE, JobError, tree, validate_inputs, validate_result
from .resources import write

UNQUANTIZED = 'model-f16.gguf'
QUANTIZED = 'model-Q4_K_M.gguf'
ADAPTER = ('adapter_model.safetensors', 'adapter_config.json')
GGUF = 'pw-custom-gguf/2'
REFERENCE = 'behavior.json'
INDEX = 'model.safetensors.index.json'
OUTPUTS = ('gguf', 'mlx', 'lora_adapter')


def checked_plan(directory):
    plan = json.loads((Path(directory) / 'plan.json').read_text())
    validate_inputs(plan)
    validate_result(plan)
    return plan


def export(directory, llama, revision, run=subprocess.run):
    directory, llama = Path(directory), Path(llama)
    plan = checked_plan(directory)
    checkout = Path(plan['checkout'])
    pipe = checkout / PIPE
    prefix = ['uv', 'run', '--project', str(pipe), '--locked', 'python']
    merged, gguf, mlx = directory / 'merged', directory / 'gguf', directory / 'mlx'
    model = Path(plan['models']) / plan['specification']['models'][plan['student']]['directory']
    args = ['--model', str(model), '--run', plan['out'], '--out', str(merged), *(['--resume'] if merged.exists() else [])]
    run([*prefix, *host_contract.script(checkout, PIPE + '/export.py', args)], check=True)
    args = ['--merge', str(merged), '--llama-cpp', str(llama),
            '--revision', revision, '--quantizer', str(llama / 'build/bin/llama-quantize'),
            '--server', str(llama / 'build/bin/llama-server'), '--out', str(gguf)]
    if gguf.exists():
        args.append('--resume')
    run([*prefix, *host_contract.script(checkout, PIPE + '/gguf_export.py', args)], check=True)
    source = mlx_source(merged)
    if not (mlx / 'conversion.json').exists():
        # The converter removes its output on failure; a killed conversion leaves a partial one.
        shutil.rmtree(mlx, ignore_errors=True)
        generation = checkout / 'scripts/generation'
        run(['uv', 'run', '--project', str(generation), '--locked', 'python',
             *host_contract.script(checkout, 'scripts/generation/convert_mlx.py',
                                   ['--source', str(merged / 'model'), '--output', str(mlx), '--source-sha256', source])], check=True)
    retain_adapter(Path(plan['out']), directory / 'lora_adapter')
    write(directory / 'export_receipt.json', describe(directory, plan, source))
    return publish(directory, revision, unquantized=True)


def mlx_source(merged):
    """The converter's source pin, derived from the merge receipt instead of rehashing the merge."""
    files = json.loads((Path(merged) / 'manifest.json').read_text())['files']
    if 'model.safetensors' in files:
        return files['model.safetensors']
    index = Path(merged) / 'model' / INDEX
    if INDEX not in files or file_sha256(index) != files[INDEX]:
        raise JobError('merged checkpoint index is not the merged one')
    names = sorted({INDEX, *json.loads(index.read_text())['weight_map'].values()})
    if any(name not in files for name in names):
        raise JobError('merged checkpoint shard is missing from the merge receipt')
    lines = ''.join('%s  %s\n' % (name, files[name]) for name in names)
    return hashlib.sha256(lines.encode()).hexdigest()


def retain_adapter(training, target):
    """Copy the adapter the training result names, so it outlives the training machine."""
    recorded = json.loads((training / 'result.json').read_text())['adapter']
    expected = {name: recorded.get(name) for name in ADAPTER}
    if target.exists():
        if tree(target) != expected:
            raise JobError('retained adapter changed')
        return
    pending = target.with_name('.pending-' + target.name)
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir()
    for name in ADAPTER:
        shutil.copyfile(training / 'adapter' / name, pending / name)
    if tree(pending) != expected:
        raise JobError('training adapter does not match its result')
    pending.rename(target)


def tensor_count(path):
    with Path(path).open('rb') as stream:
        (length,) = struct.unpack('<Q', stream.read(8))
        header = json.loads(stream.read(length))
    return len(header) - ('__metadata__' in header)


def output_files(directory, name):
    return {path: digest for path, digest in tree(directory / name).items() if path != UNQUANTIZED}


def served(manifest, name, template, reference):
    """What the host's export recorded of serving one GGUF; refused unless it rendered the
    training template as training did and, given a reference, stayed inside its tolerance."""
    record = (manifest.get('served') or {}).get(name) or {}
    rendering, behavior = record.get('rendering') or {}, record.get('behavior')
    if rendering.get('template_sha256') != template or not rendering.get('cases'):
        raise JobError('export was not served with the training chat template')
    if reference is None:
        return None
    if (not behavior or not behavior.get('agreed_tokens')
            or any(item['margin'] >= behavior['tolerance'] for item in behavior['departures'])):
        raise JobError('export has no behavioural check against its merge')
    return {key: behavior[key] for key in ('precision', 'tolerance', 'agreed_tokens', 'departures')}


def describe(directory, plan, source):
    """The receipt for one export, computed from the files it binds."""
    gguf = json.loads((directory / 'gguf/manifest.json').read_text())
    merged = json.loads((directory / 'merged/manifest.json').read_text())
    reference = file_sha256(directory / 'merged' / REFERENCE) if (directory / 'merged' / REFERENCE).is_file() else None
    if (reference is None or merged.get('behavior_sha256') != reference or gguf.get('behavior_sha256') != reference
            or merged.get('template_sha256') != plan['rendering']['template_sha256']
            or merged.get('rendering_profile') != plan['rendering']['profile']):
        raise JobError('merge does not carry the job\'s template and a behaviour reference')
    conversion = json.loads((directory / 'mlx/conversion.json').read_text())
    if conversion.get('source_sha256') != source:
        raise JobError('MLX conversion is of another checkpoint')
    weights, adapter = directory / 'mlx/model.safetensors', directory / 'lora_adapter' / ADAPTER[0]
    template = gguf['quantized']['metadata']['tokenizer.chat_template']
    return {
        'schema': 'pw-student-export/2', 'student': plan['student'],
        'merged_manifest_sha256': file_sha256(directory / 'merged/manifest.json'),
        'rendering': plan['rendering'], 'behavior_sha256': reference,
        'outputs': {
            'vulkan': {'directory': 'gguf', 'payload': QUANTIZED, 'bytes': (directory / 'gguf' / QUANTIZED).stat().st_size,
                       'sha256': gguf['quantized']['sha256'], 'tensors': len(gguf['quantized']['tensors']),
                       'quantization': 'Q4_K_M', 'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
                       'behavior': served(gguf, 'quantized', plan['rendering']['template_sha256'], reference),
                       'files': output_files(directory, 'gguf')},
            'mlx': {'directory': 'mlx', 'payload': 'model.safetensors', 'bytes': weights.stat().st_size,
                    'sha256': conversion['weights_sha256'], 'tensors': tensor_count(weights),
                    'quantization': conversion['recipe'], 'profile': conversion['profile'], 'source_sha256': source,
                    'chat_template_sha256': conversion['chat_template_sha256'], 'files': output_files(directory, 'mlx')},
            'lora_adapter': {'directory': 'lora_adapter', 'payload': ADAPTER[0], 'bytes': adapter.stat().st_size,
                             'sha256': file_sha256(adapter), 'tensors': tensor_count(adapter),
                             'files': output_files(directory, 'lora_adapter')},
        },
    }


def publish(directory, revision, unquantized):
    """Bind the served GGUF and the export receipt to their job. A remote export returns no unquantized GGUF."""
    directory = Path(directory)
    plan = checked_plan(directory)
    gguf = directory / 'gguf'
    manifest = json.loads((gguf / 'manifest.json').read_text())
    if (manifest.get('schema') != GGUF or manifest.get('llama_cpp_revision') != revision
            or manifest.get('merged_manifest_sha256') != file_sha256(directory / 'merged' / 'manifest.json')
            or (gguf / UNQUANTIZED).exists() is not unquantized
            or (unquantized and manifest.get('unquantized', {}).get('sha256') != file_sha256(gguf / UNQUANTIZED))
            or manifest.get('quantized', {}).get('sha256') != file_sha256(gguf / QUANTIZED)):
        raise JobError('export lacks a served GGUF receipt')
    receipt = json.loads((directory / 'export_receipt.json').read_text())
    if receipt != describe(directory, plan, receipt['outputs']['mlx']['source_sha256']):
        raise JobError('export receipt does not match its outputs')
    template = plan['rendering']['template_sha256']
    if any(receipt['outputs'][name]['chat_template_sha256'] != template for name in ('vulkan', 'mlx')):
        raise JobError('an export does not carry the training chat template')
    result = {'schema': 'pw-student-artifact/3', 'student': plan['student'],
              'specification_sha256': plan['specification_sha256'], 'host_commit': plan['host_commit'],
              'dry_run': plan.get('dry_run', False), 'unquantized_retained': unquantized,
              'context': plan['specification']['context'], 'files': tree(gguf), 'directory': str(gguf),
              'template_sha256': template,
              'export_receipt_sha256': file_sha256(directory / 'export_receipt.json')}
    write(directory / 'artifact.json', result)
    return result


def returned(directory):
    """True when this job's export ran on its training machine and only the served artifacts came back."""
    path = Path(directory) / 'artifact.json'
    return path.is_file() and json.loads(path.read_text()).get('unquantized_retained') is False
