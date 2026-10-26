"""Durable stage execution with reconciliation before any repeated side effect."""
import json
import os
import time
import traceback
from pathlib import Path

from .journal import CampaignFault, canonical, digest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.pending-' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Executor:
    def __init__(self, journal, backend, sleep=time.sleep):
        self.journal, self.backend, self.sleep = journal, backend, sleep

    def run(self, name, inputs):
        previous = self.journal.stage(name)
        for retry in range(3):
            current = self.journal.begin(name, inputs)
            if current['status'] in {'passed', 'skipped_dependency'}:
                self.backend.validate(name, inputs, current['output'])
                return current
            if current['status'] == 'failed':
                return current
            try:
                if previous and previous['status'] == 'running':
                    output = self.backend.reconcile(name, inputs, current['attempt'])
                else:
                    output = self.backend.execute(name, inputs, current['attempt'])
                self.backend.validate(name, inputs, output)
                return self.journal.settle(name, current['attempt'], 'passed', output=output)
            except CampaignFault as exc:
                result = self.journal.settle(name, current['attempt'], 'failed', fault=exc.record())
                if not exc.retryable or current['tries'] >= 3:
                    return result
                self.sleep(2 ** retry)
            except Exception as exc:
                diagnostic = self.journal.directory / 'diagnostics' / (current['attempt'] + '.txt')
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(diagnostic, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, 'w') as stream:
                    stream.write(''.join(traceback.format_exception(exc)))
                    stream.flush()
                    os.fsync(stream.fileno())
                return self.journal.settle(name, current['attempt'], 'failed', fault={
                    'code': 'stage_execution_unclassified', 'retryable': False,
                    'details': {'exception_type': type(exc).__name__, 'diagnostic': str(diagnostic)}})
            previous = None
        return self.journal.stage(name)

    def skip(self, name, inputs, dependency):
        current = self.journal.begin(name, inputs)
        if current['status'] == 'running':
            return self.journal.settle(name, current['attempt'], 'skipped_dependency', output={'dependency': dependency})
        return current


class ReceiptBackend:
    """Persist handler outputs before committing the controller's stage result."""
    def __init__(self, directory, handlers):
        self.directory, self.handlers = Path(directory), handlers

    def path(self, name, attempt):
        return self.directory / name / (attempt + '.json')

    def execute(self, name, inputs, attempt):
        handler = self.handlers[name]
        output = handler.execute(inputs, attempt)
        handler.validate(inputs, output)
        atomic_json(self.path(name, attempt), {'inputs_sha256': digest(inputs), 'output': output,
                                             'output_sha256': digest(output), 'attempt': attempt})
        return output

    def reconcile(self, name, inputs, attempt):
        path = self.path(name, attempt)
        if path.exists():
            record = json.loads(path.read_text())
            if (record.get('inputs_sha256') != digest(inputs) or record.get('attempt') != attempt
                    or record.get('output_sha256') != digest(record.get('output'))):
                raise CampaignFault('stage_receipt_mismatch')
            return record['output']
        output = self.handlers[name].reconcile(inputs, attempt)
        self.handlers[name].validate(inputs, output)
        atomic_json(path, {'inputs_sha256': digest(inputs), 'output': output,
                           'output_sha256': digest(output), 'attempt': attempt})
        return output

    def validate(self, name, inputs, output):
        self.handlers[name].validate(inputs, output)
