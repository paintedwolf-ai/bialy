"""Check source prerequisites before a campaign spends on collection."""
import argparse
import json
import subprocess
from pathlib import Path

from .. import host_bundle
from .execution import atomic_json
from .journal import CampaignFault


def inspect(checkout, commit, image, destination, process=subprocess.run):
    """The pinned checkout is clean at commit, and its image exports that commit's training contract."""
    checkout = Path(checkout).resolve()
    actual = process(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if actual != commit:
        raise CampaignFault('host_source_revision_mismatch', details={'expected': commit, 'actual': actual})
    dirty = process(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=normal'],
                    check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise CampaignFault('host_source_not_clean')
    try:
        contract, contract_sha256 = host_bundle.contract(image, destination, commit, process)
    except host_bundle.HostEvidenceError as exc:
        raise CampaignFault(exc.code, details=exc.details) from exc
    return {'host_commit': actual, 'contract_sha256': contract_sha256, 'sources': contract['sources']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-checkout', required=True, type=Path)
    parser.add_argument('--host-commit', required=True)
    parser.add_argument('--host-image', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {'host_checkout': str(args.host_checkout.resolve()), 'host_commit': args.host_commit, 'host_image': args.host_image}
    atomic_json(args.out / 'inputs.json', inputs)
    try:
        evidence = inspect(args.host_checkout, args.host_commit, args.host_image, args.out / 'contract')
        result = {'status': 'source_prerequisites_passed', 'evidence': evidence}
    except CampaignFault as exc:
        result = {'status': 'blocked', 'fault': exc.record()}
    except (OSError, subprocess.SubprocessError) as exc:
        result = {'status': 'blocked', 'fault': {'code': 'host_source_inspection_failed', 'retryable': False,
                                               'details': {'exception_type': type(exc).__name__}}}
    result.update(schema='pw-campaign-source-preflight/2', inputs=inputs,
                  live_host_qualified=False, collection_started=False, training_started=False)
    atomic_json(args.out / 'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'source_prerequisites_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
