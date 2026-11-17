"""The rehearsal of one tier on its training machine, with rows no model wrote.

A rehearsal trains on a handful of generated rows, one of them filled to the declared
maximum sequence length, on the `rehearsal` schedule (a checkpoint every update, a pause
after the first, a resume to the end), and then runs the whole export. It shows that the
tier fits its machine at the declared length and that every step after collection works;
it says nothing about quality.

Each row takes the path a collected row takes: a host episode bundle, the host's
candidate export, the factory's admission, the pinned tokenizer's measurement, and the
host's sealing of the splits. Only the origin differs: the conversations are generated
from fixed seeds, their source class is `synthetic`, and the quality verdict names the
bundle's operation receipt instead of a reviewer.
"""
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .. import training_rows
from ..host_bundle import Episode
from . import machines, remote_training
from .jobs import PIPE, JobError
from .resources import write
from .specification import load, serving

EPOCH = 'rehearsal'
FILES = {'train': 'train.jsonl', 'validation': 'val.jsonl', 'system_test': 'holdout.jsonl'}
# Rows by partition: one training row at the declared length beside one short row, so
# every update of the three epochs holds the long row.
PLAN = {'train': ('long', 'short'), 'validation': ('short',), 'system_test': ('short',)}
# Lines of the long row's completion: about 1,500 tokens of the 2,048 the memory estimate allows.
COMPLETION_LINES = 120
# Lines a tool result carries before the next call.
RESULT_LINES = 400
# Candidates measured in one tokenizer run while the long row is sized.
BATCH = 8
# The long row ends within this many tokens of the limit: under two generated lines.
SLACK = 64
BUNDLE_FILES = ('catalog.json', 'sessions.jsonl', 'captures.jsonl', 'operations.jsonl', 'observations.jsonl',
                'effects.jsonl', 'decisions.jsonl', 'completions.jsonl', 'source-inventory.json')
WORDS = ('alpha', 'buffer', 'cache', 'delta', 'entry', 'frame', 'graph', 'handle', 'index', 'journal', 'kernel', 'ledger',
         'margin', 'node', 'offset', 'packet', 'queue', 'record', 'segment', 'token', 'unit', 'vector', 'window', 'yield')
GUIDE = ('You are a coding agent working in a checked-out repository. Read before you change anything, '
         'keep each change small, and report what you verified.')
TOOLS = [
    {'type': 'function', 'function': {'name': 'read', 'description': 'Read a file from the repository.', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'edit', 'description': 'Replace a file with new content.', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
        'required': ['path', 'content']}}},
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def lines(seed, count):
    """`count` lines of generated source text, the same for the same seed."""
    source = random.Random(seed)
    return ['%s_%d = %s(%s, %d)  # %s %s' % (source.choice(WORDS), index, source.choice(WORDS), source.choice(WORDS),
                                            source.randrange(1000), source.choice(WORDS), source.choice(WORDS))
            for index in range(count)]


def conversation(group, count):
    """A session that reads `count` generated lines, a file a call, and then rewrites one file."""
    text = lines(group, count + COMPLETION_LINES)
    messages = [{'role': 'system', 'content': GUIDE},
                {'role': 'user', 'content': 'Session %s: read through the module, then tidy its first file.' % group}]
    for index, start in enumerate(range(0, count, RESULT_LINES)):
        identifier = 'read-%04d' % index
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': [{'id': identifier, 'type': 'function', 'function': {
            'name': 'read', 'arguments': json.dumps({'path': 'src/part_%04d.py' % index})}}]})
        messages.append({'role': 'tool', 'tool_call_id': identifier, 'content': '\n'.join(text[start:min(start + RESULT_LINES, count)]) + '\n'})
    return messages, {'path': 'src/part_0000.py', 'content': '\n'.join(text[count:]) + '\n'}


def write_bundle(root, host_commit, capture, operation, completion):
    """One settled turn as the host bundles an episode: bodies in blobs, every file bound by the manifest."""
    (root / 'blobs').mkdir(parents=True)
    blobs = []

    def blob(value):
        raw = json.dumps(value).encode()
        name = hashlib.sha256(raw).hexdigest()
        (root / 'blobs' / name).write_bytes(raw)
        blobs.append(name)
        return name

    record = dict(capture)
    for field in ('request', 'student_request', 'response'):
        record[field + '_blob'] = blob(record.pop(field))
    contract = {'schema': 'pw-training-contract/1', 'host_commit': host_commit, 'host_modified': False,
                'bundle': 'pw-training-episode-bundle/1', 'capture': 'pw-training-turn/1', 'fact': 'pw-training-fact/1',
                'files': list(BUNDLE_FILES), 'sources': [{'source': 'invocation-receipt', 'contract': 'pw-operation-receipt/1'}],
                'settlement': ['settled', 'unsettled'], 'semantic_outcome': ['success', 'failure', 'unknown'],
                'effect_state': ['none', 'known', 'unknown'], 'offer_sources': ['standing']}
    content = {'sessions.jsonl': [{'id': 'session', 'project_id': 'project', 'messages_blob': blob([])}],
               'captures.jsonl': [record], 'operations.jsonl': [operation], 'observations.jsonl': [], 'effects.jsonl': [],
               'decisions.jsonl': [], 'completions.jsonl': [dict(completion, body_blob=blob({'content': '', 'tool_calls': []}))]}
    for name, values in content.items():
        (root / name).write_text(''.join(json.dumps(value) + '\n' for value in values))
    (root / 'catalog.json').write_text(json.dumps(contract))
    (root / 'source-inventory.json').write_text(json.dumps({'delivered_paths': [], 'snapshots': {}, 'promotions': {}}))
    manifest = {'schema': 'pw-training-episode-bundle/1', 'host_commit': host_commit, 'root_session_id': 'session',
                'project_id': 'project', 'store_sha256': hashlib.sha256(root.name.encode()).hexdigest(),
                'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in BUNDLE_FILES},
                'blobs': sorted(set(blobs))}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


class Rows:
    """Generated conversations taken through the host's candidate export and the factory's admission."""

    def __init__(self, student, specification, checkout, host_commit, work, process=subprocess.run):
        self.identity = serving(specification)[student]
        self.checkout, self.host_commit, self.work, self.process = Path(checkout), host_commit, Path(work), process
        self.made = {}

    def row(self, group, count):
        """The admitted training row of the conversation that reads `count` lines."""
        name = '%s-%05d' % (group, count)
        if name in self.made:
            return self.made[name]
        messages, arguments = conversation(group, count)
        model = self.identity['model']
        wire = {'model': model, 'reasoning_effort': 'none', 'stream': True, 'tools': TOOLS, 'messages': messages}
        capture = {'schema': 'pw-training-turn/1', 'epoch': EPOCH, 'dispatches': 1, 'settled': True, 'output_id': 'output',
                   'session_id': 'session', 'message_id': 'message', 'call_id': 'request-' + name,
                   'student_provider': self.identity['provider'], 'student_kind': self.identity['kind'], 'student_model': model,
                   'request': {'generated': name}, 'student_request': wire,
                   'response': {'content': '', 'tool_calls': [{'id': 'target', 'name': 'edit', 'args': arguments}]}}
        # The generated arguments hold no character the host's canonical form escapes.
        receipt = {'id': 'receipt-' + name, 'session_id': 'session', 'message_id': 'message', 'tool': 'edit',
                   'args_sha256': digest(arguments), 'source': 'invocation-receipt', 'contract': 'pw-operation-receipt/1',
                   'invoked': True, 'settlement': 'settled', 'semantic_outcome': 'success', 'effect_state': 'known',
                   'started_at': '2026-01-01T00:00:00Z', 'tool_call_id': 'target'}
        completion = {'id': 'output', 'session_id': 'session', 'message_id': 'message', 'turn_attempt_id': 'attempt-' + name,
                      'scripted': False, 'settled_at': '2026-01-01T00:00:01Z'}
        bundle = write_bundle(self.work / 'bundles' / name, self.host_commit, capture, receipt, completion)
        episode = Episode(bundle)
        binding = self.work / 'bindings' / (name + '.json')
        binding.parent.mkdir(parents=True, exist_ok=True)
        binding.write_text(json.dumps({'epoch': EPOCH, 'host_commit': self.host_commit, 'contract_sha256': episode.contract_sha256,
                                       'student_provider': self.identity['provider'], 'student_kind': self.identity['kind'],
                                       'student_model': model}))
        exported = self.work / 'candidates' / name
        self.process([sys.executable, str(self.checkout / PIPE / 'collect.py'), '--bundle', str(bundle), '--binding', str(binding),
                      '--catalog-sha256', digest(TOOLS), '--out', str(exported)], check=True, capture_output=True)
        candidates = [json.loads(line) for line in (exported / 'candidates.jsonl').read_text().splitlines() if line.strip()]
        records = training_rows.admit(candidates, episode, {}, {'checks': []})
        if len(records) != 1 or not training_rows.is_eligible(records[0]):
            raise JobError('a generated rehearsal row was not admitted: ' + json.dumps([record['admission'] for record in records]))
        row = records[0]
        row['quality'] = {'verdict': 'pass', 'method': 'deterministic', 'evidence_ids': [receipt['id']],
                          'request_sha256': row['request_capture']['sha256'], 'response_sha256': row['response_sha256']}
        row['split_group'] = group
        self.made[name] = row
        return row


def longest(make, measure, limit):
    """The most lines whose row fits `limit` tokens. A row grows with its lines, and every
    line is more than four tokens, so the answer lies under a quarter of `limit` lines."""
    low, high = 0, limit // 4
    while high - low > 1:
        points = sorted({low + (high - low) * part // (BATCH + 1) for part in range(1, BATCH + 1)} - {low, high})
        for count, result in zip(points, measure([make(count) for count in points]), strict=True):
            if 'refused' in result:
                high = min(high, count)
            else:
                low = max(low, count)
    return low


def generate(student, specification, checkout, host_commit, measure, key, out, work, process=subprocess.run):
    """Write the sealed rehearsal splits of `student` to `out`; returns each row's token count."""
    out, work, limit = Path(out), Path(work), specification['context']['max_sequence_tokens']
    rows = Rows(student, specification, checkout, host_commit, work, process)
    out.mkdir(parents=True)

    def measured(values):
        path = work / 'measure.jsonl'
        path.write_text(''.join(canonical(row) + '\n' for row in values))
        return measure(student, path)

    approved, sources, files, report = {}, {}, [], {}
    for partition, kinds in PLAN.items():
        chosen = []
        for position, kind in enumerate(kinds):
            group = 'rehearsal-%s-%d' % (partition.replace('_', '-'), position)
            count = RESULT_LINES if kind == 'short' else longest(lambda count, group=group: rows.row(group, count), measured, limit)
            chosen.append(rows.row(group, count))
        results = measured(chosen)
        if any(type(result.get('tokens')) is not int or not 0 < result['tokens'] <= limit for result in results):
            raise JobError('a generated rehearsal row does not fit the declared sequence length')
        if partition == 'train' and max(result['tokens'] for result in results) <= limit - SLACK:
            raise JobError('no generated training row reaches the declared sequence length')
        for row in chosen:
            capture = row['source']['capture_sha256']
            sources[row['split_group']] = {'source_class': 'synthetic', 'training_allowed': True, 'parents': []}
            sources[capture] = {'source_class': 'factory_capture', 'training_allowed': True, 'parents': [row['split_group']]}
            approved[digest(row)] = {'root': row['split_group'], 'task_id': row['split_group'], 'partition': partition, 'sources': [capture]}
        path = out / FILES[partition]
        path.write_text(''.join(canonical(row) + '\n' for row in chosen))
        files.append(str(path))
        report[FILES[partition]] = [result['tokens'] for result in results]
    authority = {'run_id': EPOCH, 'host_commit': host_commit, 'lineage_sha256': digest(sorted(sources))}
    request = work / 'source-request.json'
    request.write_text(json.dumps({'authority': authority, 'sources': sources, 'approved': approved, 'files': files}))
    process([sys.executable, str(Path(checkout) / PIPE / 'admission.py'), '--request', str(request), '--key', str(key),
             '--out', str(out / 'source.json')], check=True)
    return report


def rehearse(configuration_path, student, out, *, sleep=time.sleep, clock=time.time, measure=None, process=subprocess.run):
    """Rehearse `student` on the machine the configuration names. Run again with the same
    `out`, it takes up the run it finds there; a settled rehearsal needs a new `out`."""
    from ..campaign.dataset import Measurement
    from ..campaign.sources import authority_key
    configuration = json.loads(Path(configuration_path).read_text())
    machines.preflight(configuration, student)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    splits, runs = root / 'splits', root / 'runs'
    key = authority_key(root)
    if not (root / 'rows.json').exists():
        # Rows half-written by a stopped controller are generated again from the same seeds.
        for partial in (splits, root / 'rows'):
            shutil.rmtree(partial, ignore_errors=True)
        specification = load()
        measure = measure or Measurement(configuration['host_checkout'], configuration['models'], specification)
        report = generate(student, specification, configuration['host_checkout'], configuration['host_commit'], measure, key,
                          splits, root / 'rows', process)
        write(root / 'rows.json', {'student': student, 'tokens': report})
    found = sorted(runs.glob(student + '-*'))
    if len(found) > 1:
        raise JobError('more than one rehearsal run is under ' + str(runs))
    return remote_training.train(configuration_path, student, splits, key, runs, resume=found[0] if found else None,
                                 schedule='rehearsal', sleep=sleep, clock=clock)
