"""Per-tier go/no-go after a campaign, and the release ledger of what was published.

A tier is published only when its export receipt still binds its files, its MLX and
Vulkan bundles each carry a load probe the host's native runtime wrote for this export,
and its outcome under the frozen evaluation policy is a pass. A probe counts by what it
binds: the export receipt, the bundle's artifact, the behaviour reference, the host
commit the tier was trained on and the device it ran on, and tokens that stay inside
the reference's tolerance. Every other tier is recorded unpublished with the conditions
it missed.
"""
import json
import re
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
# Where a host checkout states which tiers each backend of its native runtime loads.
NATIVE_TIERS = 'scripts/generation/native_tiers.json'


def unpublishable(checkout, students):
    """The students a campaign on this host checkout cannot publish, each with the bundles its
    native runtime does not load: no load probe of those can pass."""
    path = Path(checkout) / NATIVE_TIERS
    declared = json.loads(path.read_text()) if path.is_file() else {}
    blocked = {student: [bundle for bundle in BUNDLES if student not in declared.get(bundle, [])] for student in students}
    return {student: bundles for student, bundles in blocked.items() if bundles}


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


def reference_prompt(job, name):
    reference = json.loads((job / 'merged' / exporting.REFERENCE).read_text())
    return next((prompt for prompt in reference['prompts'] if prompt['name'] == name), None)


def probe_fault(probe, bundle, artifact, receipt, job):
    """Why one probe receipt does not qualify this export's bundle; None when it does."""
    if probe.get('passed') is not True or probe.get('failure'):
        return (probe.get('failure') or {}).get('code') or 'failed'
    device = probe.get('device') or {}
    if (probe.get('export_receipt_sha256') != artifact['export_receipt_sha256']
            or probe.get('artifact_sha256') != receipt['outputs'][bundle]['sha256']
            or probe.get('behavior_sha256') != receipt['behavior_sha256']
            or probe.get('host_commit') != artifact['host_commit']):
        return 'unbound'
    if (device.get('backend') != bundle or not device.get('physical_device')
            or not re.fullmatch(r'[0-9a-f]{64}', str(probe.get('manifest_sha256')))):
        return 'device_unstated'
    prompt, behavior = reference_prompt(job, probe.get('prompt')), probe.get('behavior') or {}
    tokens = probe.get('generated_tokens') or []
    if prompt is None or not tokens or type(behavior.get('tolerance')) not in (int, float):
        return 'no_generation'
    wanted = prompt['tokens']
    shared = min(len(wanted), len(tokens))
    agreed = next((index for index in range(shared) if wanted[index] != tokens[index]), shared)
    if agreed == 0 or (agreed < len(wanted) and prompt['margins'][agreed] >= behavior['tolerance']):
        return 'behavior_mismatch'
    return None


def probed(student, artifact, probes):
    """For each bundle, the manifest a qualifying load probe of this export loaded and the
    device that ran it, and the bundles without one with the reason their probes gave."""
    job = Path(artifact['directory']).parent
    receipt = json.loads((job / 'export_receipt.json').read_text())
    manifests, failed = {}, []
    for bundle in BUNDLES:
        offered = [probe for probe in probes if probe.get('schema') == PROBE and probe.get('student') == student
                   and probe.get('bundle') == bundle]
        faults = {id(probe): probe_fault(probe, bundle, artifact, receipt, job) for probe in offered}
        qualified = [probe for probe in offered if faults[id(probe)] is None]
        if qualified:
            manifests[bundle] = {'manifest_sha256': qualified[0]['manifest_sha256'], 'device': qualified[0]['device']}
        else:
            failed.append('load_probe_%s_%s' % (bundle, min(faults.values()) if offered else 'missing'))
    return manifests, failed


def tier_decision(student, outcome, probes):
    artifact = outcome.get('artifact')
    failed = export_failures(artifact)
    # A probe can only be bound to an export whose receipt still holds.
    manifests, unprobed = ({}, []) if failed else probed(student, artifact, probes)
    failed += unprobed
    if outcome.get('execution_status') != 'complete' or outcome.get('behavioral_status') != 'pass':
        failed.append('behavioral_%s' % outcome.get('behavioral_status', 'not_measured'))
    return {'publish': not failed, 'failed': failed,
            'export_receipt_sha256': (artifact or {}).get('export_receipt_sha256'), 'probes': manifests}


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
            if any(bundles[name]['manifest_sha256'] != tier['probes'].get(name, {}).get('manifest_sha256') for name in BUNDLES):
                raise CampaignFault('publication_not_probed', details={'student': student})
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
