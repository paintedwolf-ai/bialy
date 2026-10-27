"""Train-to-export transitions with checked host parity receipts."""
import json
import subprocess
from pathlib import Path

from ..provenance import file_sha256
from .jobs import JobError, tree, validate_inputs, validate_result
from .resources import write

UNQUANTIZED = 'model-f16.gguf'


def export(directory, llama, revision, run=subprocess.run):
    directory, llama = Path(directory), Path(llama)
    plan = json.loads((directory / 'plan.json').read_text())
    validate_inputs(plan)
    validate_result(plan)
    pipe = Path(plan['checkout']) / 'scripts/bialy/custom_model'
    prefix = ['uv', 'run', '--project', str(pipe), '--locked', 'python']
    merged, gguf = directory / 'merged', directory / 'gguf'
    model = Path(plan['models']) / plan['specification']['models'][plan['student']]['directory']
    args = [*prefix, str(pipe / 'export.py'), '--model', str(model), '--run', plan['out'], '--out', str(merged)]
    if merged.exists():
        args.append('--resume')
    run(args, check=True)
    args = [*prefix, str(pipe / 'gguf_export.py'), '--merge', str(merged), '--llama-cpp', str(llama),
            '--revision', revision, '--quantizer', str(llama / 'build/bin/llama-quantize'),
            '--loader', str(llama / 'build/bin/llama-completion'), '--out', str(gguf)]
    if gguf.exists():
        args.append('--resume')
    run(args, check=True)
    return publish(directory, revision, unquantized=True)


def publish(directory, revision, unquantized):
    """Bind the served GGUF to its job. A remote export returns only the quantized file and receipts."""
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text())
    validate_inputs(plan)
    validate_result(plan)
    gguf = directory / 'gguf'
    manifest = json.loads((gguf / 'manifest.json').read_text())
    if (manifest.get('schema') != 'pw-custom-gguf/1' or manifest.get('load_probe') != 'passed'
            or manifest.get('llama_cpp_revision') != revision
            or manifest.get('merged_manifest_sha256') != file_sha256(directory / 'merged' / 'manifest.json')
            or (gguf / UNQUANTIZED).exists() is not unquantized
            or (unquantized and manifest.get('unquantized', {}).get('sha256') != file_sha256(gguf / UNQUANTIZED))
            or manifest.get('quantized', {}).get('sha256') != file_sha256(gguf / 'model-Q4_K_M.gguf')):
        raise JobError('export lacks a load parity receipt')
    result = {'schema': 'pw-student-artifact/1', 'student': plan['student'],
              'specification_sha256': plan['specification_sha256'], 'host_commit': plan['host_commit'],
              'dry_run': plan.get('dry_run', False), 'unquantized_retained': unquantized,
              'context': plan['specification']['context'], 'files': tree(gguf), 'directory': str(gguf)}
    write(directory / 'artifact.json', result)
    return result


def returned(directory):
    """True when this job's export ran on its training machine and only the served artifact came back."""
    path = Path(directory) / 'artifact.json'
    return path.is_file() and json.loads(path.read_text()).get('unquantized_retained') is False
