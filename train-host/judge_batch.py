"""Run one resumable judging job from a saved manifest.

Usage: PYTHONPATH=src python train-host/judge_batch.py MANIFEST JOB
Credentials are supplied through the model configuration's environment variable.
Run under detach.py locally or jobctl.sh remotely. Paths are relative to the
manifest. Each job owns an exclusive lease and writes an atomic result receipt.
"""
import argparse
import fcntl
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bialy import config, judge


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(manifest_path, name):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    job = manifest['jobs'][name]
    root = manifest_path.parent
    corpus = root / manifest['corpus']
    source, target = root / job['rows'], root / job['out']
    target.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('BIALY_USAGE_JSONL', str(target) + '.usage.jsonl')
    with open(str(target) + '.lease', 'a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {'job': name, 'started': now(), 'factory_commit': manifest['factory_commit'],
                  'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        try:
            result['report'] = judge.run(config.load(), json.loads(corpus.read_text()), str(source), str(target),
                                         workers=job.get('workers', 20), second=1.0, units=tuple(job.get('units', ['skills'])))
            result['success'] = result['report']['failed'] == 0
        except Exception:
            result['success'] = False
            result['error'] = traceback.format_exc()
        result['finished'] = now()
        temp = Path(str(target) + '.result.tmp')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(temp, str(target) + '.result.json')
        print(json.dumps(result), flush=True)
        return result['success']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest')
    ap.add_argument('job')
    args = ap.parse_args()
    if not execute(args.manifest, args.job):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
