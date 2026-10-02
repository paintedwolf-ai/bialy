"""Write or verify a SHA256 manifest of regular files without following symlinks.

Usage: manifest.py write ROOT MANIFEST | manifest.py verify ROOT MANIFEST
Manifest paths are relative to ROOT; the manifest itself is excluded.
Verification rejects missing, changed, and extra files.
"""
import argparse
import hashlib
import json
from pathlib import Path


def inventory(root, manifest):
    entries = {}
    for path in sorted(root.rglob('*')):
        if path == manifest or not path.is_file() or path.is_symlink():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                digest.update(chunk)
        entries[str(path.relative_to(root))] = digest.hexdigest()
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['write', 'verify'])
    ap.add_argument('root', type=Path)
    ap.add_argument('manifest', type=Path)
    args = ap.parse_args()
    root, manifest = args.root.resolve(), args.manifest.resolve()
    actual = inventory(root, manifest)
    if args.mode == 'write':
        manifest.write_text(json.dumps(actual, indent=2) + '\n')
        print(f'manifest: {len(actual)} files')
        return
    expected = json.loads(manifest.read_text())
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    changed = [name for name in expected.keys() & actual.keys() if expected[name] != actual[name]]
    print(json.dumps({'files': len(actual), 'missing': missing, 'extra': extra, 'changed': changed}))
    if missing or extra or changed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
