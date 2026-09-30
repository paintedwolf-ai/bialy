"""Recover worker decision inputs from immutable archived job records.

Usage: recover_worker_requests.py ROWS_JSONL ARCHIVE_ROOT OUT_JSONL REPORT_JSON
Only matches the row's opening message, child session, and parent session to the
recorded worker job. Never parses the rendered assignment prose. Refuses archives
with WAL files because immutable SQLite reads would miss uncheckpointed records.
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


def recover(rows, root):
    wanted = {r['opening_message_id']: r for r in rows if r['host'] == 'worker' and not r['partial']}
    recovered, conflicts, errors = {}, [], []
    for path in sorted(Path(root).glob('**/store.db')):
        if path.with_name(path.name + '-wal').exists():
            errors.append({'path': str(path), 'error': 'WAL present; need a consistent snapshot'})
            continue
        try:
            conn = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
            for msg, child, parent, job, prompt, brief, leg in conn.execute('''
                SELECT m.id, j.child_session_id, j.parent_session_id, j.id, j.prompt, j.brief, j.leg_id
                FROM messages m JOIN worker_jobs j ON m.worker_job_id = j.id
                WHERE m.session_id = j.child_session_id AND m.role = 'user'
            '''):
                prompt = prompt if leg else brief
                row = wanted.get(msg)
                if row is None or child != row['session'] or parent != row['root_session'] or not prompt.strip():
                    continue
                text = prompt.strip()[:900].strip()
                if msg in recovered and recovered[msg]['text'] != text:
                    conflicts.append(msg)
                    continue
                recovered[msg] = {'text': text, 'job': job, 'db': str(path),
                                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
            conn.close()
        except sqlite3.Error as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    if conflicts:
        raise ValueError('conflicting archived job prompts: ' + ', '.join(sorted(set(conflicts))))
    output = []
    for row in rows:
        found = recovered.get(row['opening_message_id'])
        if found is None:
            continue
        row = json.loads(json.dumps(row))
        row['state']['user'] = found['text']
        row.setdefault('meta', {})['worker_request_recovery'] = {k: v for k, v in found.items() if k != 'text'}
        output.append(row)
    return output, {'wanted': len(wanted), 'recovered': len(output), 'missing': sorted(set(wanted) - set(recovered)), 'errors': errors}


def main():
    source, root, output, report = sys.argv[1:]
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    fixed, result = recover(rows, root)
    Path(output).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in fixed))
    Path(report).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'wanted': result['wanted'], 'recovered': result['recovered'], 'errors': len(result['errors'])}))


if __name__ == '__main__':
    main()
