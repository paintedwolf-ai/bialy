"""Check source prerequisites before a campaign spends on collection."""
import argparse
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

from .execution import atomic_json
from .journal import CampaignFault

COMMAND_COLUMNS = {
    'invocation_programs': {'id', 'parent_receipt_id', 'state', 'status_code', 'settled_at',
                            'progress_state', 'source_exact', 'syntax_validated'},
    'invocation_program_executions': {'id', 'program_id', 'ordinal', 'state', 'status_code',
                                     'effect_certainty', 'settled_at'},
}


def command_schema(schema):
    with sqlite3.connect(':memory:') as database:
        try:
            database.executescript(schema)
        except sqlite3.Error as exc:
            raise CampaignFault('host_schema_unreadable', details={'exception_type': type(exc).__name__}) from exc
        missing = {}
        for table, required in COMMAND_COLUMNS.items():
            columns = {row[1] for row in database.execute('PRAGMA table_info(' + table + ')')}
            if required - columns:
                missing[table] = sorted(required - columns)
    return {'schema_sha256': hashlib.sha256(schema.encode()).hexdigest(), 'missing_columns': missing}


def inspect(checkout, commit, process=subprocess.run):
    checkout = Path(checkout).resolve()
    actual = process(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if actual != commit:
        raise CampaignFault('host_source_revision_mismatch', details={'expected': commit, 'actual': actual})
    dirty = process(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=normal'],
                    check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise CampaignFault('host_source_not_clean')
    result = command_schema((checkout / 'lycaon/internal/db/schema.sql').read_text())
    if result['missing_columns']:
        raise CampaignFault('host_command_receipts_unavailable', details={'host_commit': actual, **result})
    return {'host_commit': actual, **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-checkout', required=True, type=Path)
    parser.add_argument('--host-commit', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {'host_checkout': str(args.host_checkout.resolve()), 'host_commit': args.host_commit}
    atomic_json(args.out / 'inputs.json', inputs)
    try:
        evidence = inspect(args.host_checkout, args.host_commit)
        result = {'status': 'source_prerequisites_passed', 'evidence': evidence}
    except CampaignFault as exc:
        result = {'status': 'blocked', 'fault': exc.record()}
    except (OSError, subprocess.SubprocessError) as exc:
        result = {'status': 'blocked', 'fault': {'code': 'host_source_inspection_failed', 'retryable': False,
                                               'details': {'exception_type': type(exc).__name__}}}
    result.update(schema='pw-campaign-source-preflight/1', inputs=inputs,
                  live_host_qualified=False, collection_started=False, training_started=False)
    atomic_json(args.out / 'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'source_prerequisites_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
