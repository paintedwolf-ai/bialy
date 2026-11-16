"""Per-tier go/no-go after a campaign, and the release ledger of what was published.

A tier is published only when its export receipt still binds its files, its MLX and
Vulkan bundles each passed a load probe in the host's native runtime, and its outcome
under the frozen evaluation policy is a pass. Every other tier is recorded unpublished
with the conditions it missed.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from ..provenance import file_sha256
from ..student_host import exporting
from .execution import atomic_json
from .journal import CampaignFault

DECISION = 'pw-tier-decision/1'
PROBE = 'pw-load-probe/1'
LEDGER = 'pw-student-release/1'
BUNDLES = ('mlx', 'vulkan')


def export_failures(artifact):
    """Why the tier's export no longer matches its receipt; empty when it does."""
    if not artifact:
        return ['no_artifact']
    job = Path(artifact['directory']).parent
    receipt = job / 'export_receipt.json'
    if not receipt.is_file() or file_sha256(receipt) != artifact.get('export_receipt_sha256'):
        return ['export_receipt_changed']
    outputs = json.loads(receipt.read_text())['outputs']
    return ['export_%s_changed' % name for name, output in sorted(outputs.items())
            if exporting.output_files(job, output['directory']) != output['files']]


def probe_failures(student, artifact, probes):
    """The bundles without a passed load probe of this export in the host's native runtime."""
    receipt = (artifact or {}).get('export_receipt_sha256')
    passed = {probe['bundle'] for probe in probes
              if probe.get('schema') == PROBE and probe.get('student') == student and probe.get('passed') is True
              and probe.get('export_receipt_sha256') == receipt}
    return ['load_probe_%s' % bundle for bundle in BUNDLES if bundle not in passed]


def tier_decision(student, outcome, probes):
    failed = export_failures(outcome.get('artifact')) + probe_failures(student, outcome.get('artifact'), probes)
    if outcome.get('execution_status') != 'complete' or outcome.get('behavioral_status') != 'pass':
        failed.append('behavioral_%s' % outcome.get('behavioral_status', 'not_measured'))
    return {'publish': not failed, 'failed': failed,
            'export_receipt_sha256': (outcome.get('artifact') or {}).get('export_receipt_sha256')}


def decide(run_directory, probes, decided_by, clock=None):
    """Write the run's decision once; a second decision must be the same one."""
    run_directory = Path(run_directory)
    result = json.loads((run_directory / 'result.json').read_text())
    if result.get('dry_run') or result.get('profile') != 'full':
        raise CampaignFault('decision_needs_full_campaign', details={'profile': result.get('profile')})
    if not decided_by:
        raise CampaignFault('decision_needs_a_decider')
    tiers = {student: tier_decision(student, outcome, probes) for student, outcome in sorted(result['students'].items())}
    path = run_directory / 'decision.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['tiers'] != tiers or existing['decided_by'] != decided_by:
            raise CampaignFault('decision_changed')
        return existing
    decision = {'schema': DECISION, 'journal_sha256': result['journal_sha256'], 'decided_by': decided_by,
                'decided_at': (clock or (lambda: datetime.now(timezone.utc)))().isoformat(), 'tiers': tiers}
    atomic_json(path, decision)
    return decision


def record(run_directory, ledger_directory, campaign, publications):
    """The release ledger entry: each published tier's bundle revisions and manifest digests,
    and each unpublished tier with the conditions it missed. Only decided tiers may appear."""
    decision = json.loads((Path(run_directory) / 'decision.json').read_text())
    published, unpublished = {}, {}
    for student, tier in decision['tiers'].items():
        bundles = publications.get(student)
        if tier['publish']:
            if not bundles or set(bundles) != set(BUNDLES) or any(
                    set(bundle) != {'revision', 'manifest_sha256'} for bundle in bundles.values()):
                raise CampaignFault('publication_incomplete', details={'student': student})
            published[student] = {'export_receipt_sha256': tier['export_receipt_sha256'], 'bundles': bundles}
        else:
            if bundles:
                raise CampaignFault('publication_not_decided', details={'student': student})
            unpublished[student] = tier['failed']
    entry = {'schema': LEDGER, 'campaign': campaign, 'decision_sha256': file_sha256(Path(run_directory) / 'decision.json'),
             'decided_by': decision['decided_by'], 'published': published, 'unpublished': unpublished}
    path = Path(ledger_directory) / ('students-%s.json' % campaign)
    if path.exists() and json.loads(path.read_text()) != entry:
        raise CampaignFault('release_ledger_changed', details={'path': str(path)})
    atomic_json(path, entry)
    return entry
