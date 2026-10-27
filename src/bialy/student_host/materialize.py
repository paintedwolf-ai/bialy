"""Materialize pinned model files and preserve their verified source identity."""
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from .jobs import JobError, tree
from .resources import write
from .specification import identity


def models(specification, root, run=subprocess.run, students=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selected = list(specification['models']) if students is None else list(students)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(specification['models']):
        raise JobError('model selection must name declared students')
    results = {}
    for name in selected:
        model = specification['models'][name]
        target = root / model['directory']
        receipt_path = root / (model['directory'] + '.source.json')
        expected = {'repository': model['repository'], 'revision': model['revision']}
        if target.exists():
            if not receipt_path.is_file():
                raise JobError('existing checkpoint has no source receipt')
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('source') != expected or receipt.get('files') != tree(target):
                raise JobError('checkpoint source or content changed')
        else:
            staging = root / ('.download-' + name + '-' + uuid.uuid4().hex)
            staging.mkdir()
            run(['hf', 'download', model['repository'], '--revision', model['revision'], '--local-dir', str(staging)], check=True)
            cache = staging / '.cache'
            if cache.exists():
                shutil.rmtree(cache)
            receipt = {'schema': 'pw-student-source/1', 'source': expected, 'files': tree(staging)}
            write(receipt_path, receipt)
            os.rename(staging, target)
        results[name] = {'directory': str(target), 'source_sha256': identity(receipt)}
    return results
