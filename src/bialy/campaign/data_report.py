"""data-report.{json,md}: the corpus ledger of every student, reproducible from sealed inputs.

It holds counts and digests only, never source text or credentials.
"""
from pathlib import Path

from .execution import atomic_json
from .journal import canonical, digest

SCHEMA = 'pw-data-report/1'


def build(binding, ledgers, status, deficits):
    body = {'schema': SCHEMA, 'binding_sha256': digest(binding), 'host_commit': binding['host_commit'],
            'selector': binding['selector'], 'catalog_sha256': binding['catalog_sha256'], 'data_status': status,
            'deficits': deficits, 'students': {student: ledgers[student] for student in sorted(ledgers)}}
    return {**body, 'report_sha256': digest(body)}


def markdown(report):
    lines = ['# Data report', '', 'Data status: %s. Host %s; selector %s; catalog %s.' % (
        report['data_status'], report['host_commit'], report['selector'].get('policy') or 'conservative reference',
        report['catalog_sha256']), '']
    for student, ledger in report['students'].items():
        lines += ['## %s' % student, '', '| Partition | Groups | Repositories | Episodes | Encodings | Categories | Tokens p50 / max |',
                  '|---|---|---|---|---|---|---|']
        for partition, counts in ledger['independent'].items():
            tokens = ledger['tokens'][partition]
            categories = ', '.join('%s %d' % item for item in ledger['categories'][partition].items()) or 'none'
            lines.append('| %s | %d | %d | %d | %d | %s | %s / %s |' % (
                partition, counts['groups'], counts['repositories'], counts['episodes'], counts['encodings'],
                categories, tokens['p50'], tokens['max']))
        strata = ', '.join('%s %d' % item for item in ledger['strata'].items())
        exercise = ledger['exercise']
        lines += ['', 'Selector strata: %s.' % strata,
                  'Compacted episodes %d; worker rows %d across %d episodes.' % (
                      exercise['compacted_episodes'], exercise['worker_rows'], exercise['worker_episodes']),
                  'Exclusions: %s.' % (', '.join('%s %d' % item for item in ledger['exclusions'].items()) or 'none'),
                  'Infrastructure faults: %s.' % (', '.join('%s %d' % item for item in ledger['infrastructure_faults'].items()) or 'none'), '']
    for student, partitions in sorted(report['deficits'].items()):
        for partition, short in sorted(partitions.items()):
            lines.append('Deficit for %s in %s: %d episodes short; missing focuses: %s.' % (
                student, partition, short['episodes_short'], ', '.join(short['focuses_missing']) or 'none'))
    return '\n'.join(lines).rstrip() + '\n'


def write(directory, report):
    directory = Path(directory)
    atomic_json(directory / 'data-report.json', report)
    (directory / 'data-report.md').write_text(markdown(report))
    return canonical(report)
