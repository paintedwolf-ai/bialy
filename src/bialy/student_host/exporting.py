"""Train-to-export transitions with checked host parity receipts."""
import json
import subprocess
from pathlib import Path

from .jobs import JobError, tree, validate_inputs, validate_result
from .resources import write


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
    manifest = json.loads((gguf / 'manifest.json').read_text())
    if not manifest.get('load_probe'):
        raise JobError('export lacks a load parity receipt')
    result = {'schema': 'pw-student-artifact/1', 'student': plan['student'],
              'specification_sha256': plan['specification_sha256'], 'host_commit': plan['host_commit'],
              'context': plan['specification']['context'], 'files': tree(gguf), 'directory': str(gguf)}
    write(directory / 'artifact.json', result)
    return result
