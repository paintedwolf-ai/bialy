"""Export typed evidence from a closed capture through its pinned host image."""
import json
import subprocess

from .execution import atomic_json
from .journal import CampaignFault


def read(shard, project, session, image, destination, process=subprocess.run):
    if destination.exists():
        return json.loads(destination.read_text())
    result = process(['docker', 'run', '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '-v', str(shard.resolve()) + ':/capture:ro', '-v', str(project.resolve()) + ':/project:ro',
        '--entrypoint', '/opt/bialy/bin/lycaon-debug', image, 'eval', 'evidence',
        '--capture', '/capture', '--session', session, '--project', '/project'],
        check=False, capture_output=True, text=True)
    if result.returncode:
        diagnostic = destination.with_name('evidence-reader-failure.json')
        atomic_json(diagnostic, {'exit_status': result.returncode, 'stderr': result.stderr})
        diagnostic.chmod(0o600)
        raise CampaignFault('host_evidence_export_failed', details={'exit_status': result.returncode,
                            'diagnostic': str(diagnostic)})
    try:
        facts = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignFault('host_evidence_invalid_json') from exc
    if not isinstance(facts, dict):
        raise CampaignFault('host_evidence_invalid_shape')
    atomic_json(destination, facts)
    return facts
