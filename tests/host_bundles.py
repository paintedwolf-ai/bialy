"""Write host episode bundles and stand in for the pinned image's lycaon-debug."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

FILES = ['catalog.json', 'sessions.jsonl', 'captures.jsonl', 'operations.jsonl', 'observations.jsonl',
         'effects.jsonl', 'decisions.jsonl', 'completions.jsonl', 'source-inventory.json']


def contract(commit='c' * 40, **changes):
    return {'schema': 'pw-training-contract/1', 'host_commit': commit, 'host_modified': False,
            'bundle': 'pw-training-episode-bundle/1', 'capture': 'pw-training-turn/1', 'fact': 'pw-training-fact/1',
            'files': FILES, 'sources': [{'source': 'invocation-receipt', 'contract': 'pw-operation-receipt/1'}],
            'settlement': ['settled', 'unsettled'], 'semantic_outcome': ['success', 'failure', 'unknown'],
            'effect_state': ['none', 'known', 'unknown'], 'offer_sources': ['standing'], **changes}


def operation(identifier, call='', parent='', tool='read', session='root', invoked=True, settlement='settled',
              outcome='success', effects='known'):
    value = {'id': identifier, 'session_id': session, 'tool': tool, 'source': 'invocation-receipt',
             'contract': 'pw-operation-receipt/1', 'invoked': invoked, 'settlement': settlement,
             'semantic_outcome': outcome, 'effect_state': effects, 'started_at': '2026-11-12T10:00:00Z'}
    if call:
        value['tool_call_id'] = call
    if parent:
        value['parent_id'] = parent
    return value


def write_episode(root, sessions=None, captures=(), operations=(), observations=(), effects=(), commit='c' * 40, store='s' * 64):
    """A bundle as the host writes it: transcripts and capture bodies in blobs, every file bound."""
    root = Path(root)
    (root / 'blobs').mkdir(parents=True)
    blobs = []

    def blob(value):
        raw = json.dumps(value).encode()
        name = hashlib.sha256(raw).hexdigest()
        (root / 'blobs' / name).write_bytes(raw)
        blobs.append(name)
        return name

    session_lines = []
    for session in sessions or [{'id': 'root', 'messages': []}]:
        session = dict(session)
        session_lines.append({'id': session['id'], 'parent_id': session.get('parent_id', ''), 'project_id': 'project',
                              'messages_blob': blob(session.pop('messages', []))})
    capture_lines = []
    for record in captures:
        record = dict(record)
        for field in ('request', 'student_request', 'response'):
            if field in record:
                record[field + '_blob'] = blob(record.pop(field))
        capture_lines.append(record)
    lines = {'sessions.jsonl': session_lines, 'captures.jsonl': capture_lines, 'operations.jsonl': list(operations),
             'observations.jsonl': list(observations), 'effects.jsonl': list(effects), 'decisions.jsonl': [],
             'completions.jsonl': []}
    for name, values in lines.items():
        (root / name).write_text(''.join(json.dumps(value) + '\n' for value in values))
    (root / 'catalog.json').write_text(json.dumps(contract(commit)))
    (root / 'source-inventory.json').write_text('{}')
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}
    manifest = {'schema': 'pw-training-episode-bundle/1', 'host_commit': commit, 'root_session_id': session_lines[0]['id'],
                'project_id': 'project', 'store_sha256': store, 'files': files, 'blobs': sorted(set(blobs))}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


class Image:
    """Answers `docker run ... lycaon-debug eval` by writing into the mounted output."""

    def __init__(self, commit='c' * 40, fail=None, **episode):
        self.commit, self.fail, self.episode, self.calls = commit, fail, episode, []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        mounts = {spec.split(':')[1]: spec.split(':')[0] for flag, spec in zip(command, command[1:], strict=False) if flag == '-v'}
        arguments = command[command.index('eval') + 1:]
        if self.fail == arguments[0]:
            return SimpleNamespace(returncode=3, stdout='', stderr='refused')
        out = Path(mounts['/out']) / arguments[arguments.index('--out') + 1].removeprefix('/out/')
        if arguments[0] == 'training-contract':
            out.mkdir()
            (out / 'contract.json').write_text(json.dumps(contract(self.commit)))
        else:
            session = arguments[arguments.index('--session') + 1]
            sessions = self.episode.get('sessions') or [{'id': session, 'messages': []}]
            write_episode(out, **dict(self.episode, sessions=sessions), commit=self.commit)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
