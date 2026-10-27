"""Reuse verified episode evidence in a separately bound automation rehearsal."""
import json
import os
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import quote

from ..provenance import file_sha256
from ..student_host.specification import identity, load
from .episodes import Episodes
from .journal import CampaignFault
from .lineage import Lineage
from .sources import seal_dataset

TABLES = ('tasks', 'lineage_freeze', 'task_sessions', 'episodes', 'episode_sessions', 'judge_controls')
MAX_TOKENS = 10240


def read_source(directory):
    path = Path(directory) / 'campaign.sqlite'
    connection = sqlite3.connect('file:' + quote(str(path.resolve()), safe='/') + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def qualified(source, name):
    row = source.execute('SELECT status,output FROM stages WHERE name=?', (name,)).fetchone()
    if row is None or row['status'] != 'passed':
        raise CampaignFault('dry_run_source_stage_unqualified', details={'stage': name})
    return json.loads(row['output'])


def import_evidence(journal, directory):
    lineage = Lineage(journal)
    Episodes(journal, lineage)
    with read_source(directory) as source:
        qualified(source, 'dataset_pilot')
        with journal.db:
            for table in TABLES:
                rows = list(source.execute('SELECT * FROM ' + table))
                for row in rows:
                    journal.db.execute('INSERT OR IGNORE INTO ' + table + ' VALUES('
                                       + ','.join('?' for _ in row) + ')', tuple(row))
                retained = {tuple(row) for row in journal.db.execute('SELECT * FROM ' + table)}
                if any(tuple(row) not in retained for row in rows):
                    raise CampaignFault('dry_run_source_evidence_changed')
    lineage.freeze()
    return lineage


class Data:
    def __init__(self, journal, lineage, source, configuration, process=subprocess.run):
        self.journal, self.lineage, self.source = journal, lineage, Path(source)
        self.configuration, self.process = configuration, process

    def execute(self, inputs, attempt):
        cfg = self.configuration
        with read_source(self.source) as source:
            data = qualified(source, 'dataset_pilot')['students']['e4b']
        original = Path(data['directory'])
        if any(file_sha256(original / name) != checksum for name, checksum in data['files'].items()):
            raise CampaignFault('dry_run_source_dataset_changed')
        target = self.journal.directory / 'datasets' / attempt / 'e4b'
        selector = Path(__file__).resolve().parents[1] / 'student_host/select_rows.py'
        pipe = Path(cfg['checkout']) / 'scripts/bialy/custom_model'
        environment = dict(os.environ, PW_TRAINING_SOURCE_KEY=str(self.source / 'source.key'))
        with (self.journal.directory / ('selection-' + attempt + '.log')).open('a') as log:
            self.process(['uv', 'run', '--project', str(pipe), '--locked', 'python', str(selector),
                          '--checkout', cfg['checkout'], '--model', str(Path(cfg['models']) / 'gemma-4-E4B-it'),
                          '--splits', str(original), '--out', str(target), '--max-tokens', str(MAX_TOKENS)],
                         env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
        names = ('train.jsonl', 'val.jsonl', 'holdout.jsonl')
        files = [target / name for name in names]
        manifests = [json.loads(path.read_text())['manifest']
                     for path in self.source.glob('trials/*/exports/e4b/reviewed.json')]
        seal_dataset(self.journal, self.lineage, files, manifests, cfg['checkout'], cfg['host_commit'])
        return {'students': {'e4b': {'directory': str(target), 'specification_sha256': identity(load()),
                'files': {name: file_sha256(target / name) for name in (*names, 'source.json')}}},
                'selection': json.loads((target / 'selection.json').read_text()), 'max_example_tokens': MAX_TOKENS}

    def reconcile(self, inputs, attempt):
        raise CampaignFault('dry_run_data_outcome_unknown')

    def validate(self, inputs, output):
        item = output['students']['e4b']
        if (output['max_example_tokens'] != MAX_TOKENS or item['specification_sha256'] != identity(load())
                or any(file_sha256(Path(item['directory']) / name) != value for name, value in item['files'].items())):
            raise CampaignFault('dry_run_dataset_changed')
