"""Issue training permits from frozen task lineage and verified host exports."""
import json
import os
import subprocess
import sys
from pathlib import Path

from ..provenance import file_sha256
from .episodes import Episodes
from .execution import atomic_json
from .journal import CampaignFault, digest


def authority_key(directory):
    path = Path(directory) / 'source.key'
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.is_symlink() or path.stat().st_mode & 0o077 or len(path.read_bytes()) < 32:
            raise CampaignFault('source_authority_key_invalid') from None
    else:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(os.urandom(32))
            stream.flush()
            os.fsync(stream.fileno())
    return path


def seal_dataset(journal, lineage, files, export_manifest, checkout, host_commit, run=subprocess.run):
    exported = json.loads(Path(export_manifest).read_text())
    if exported.get('schema') != 'pw-receipt-export/1' or exported.get('identity', {}).get('host_commit') != host_commit:
        raise CampaignFault('host_export_identity_mismatch')
    for name, checksum in exported['inputs'].items():
        if file_sha256(Path(name)) != checksum:
            raise CampaignFault('host_export_input_changed')
    candidates_path = Path(export_manifest).parent / 'sft.receipts.jsonl'
    if file_sha256(candidates_path) != exported.get('rows_sha256'):
        raise CampaignFault('host_export_rows_changed')
    candidates = {digest(json.loads(line)) for line in candidates_path.read_text().splitlines() if line.strip()}
    permitted_hashes = set(exported['inputs'].values())
    sources, approved = {}, {}
    episodes = Episodes(journal, lineage)
    for path in map(Path, files):
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            candidate = {key: value for key, value in row.items() if key not in {'meta', 'split_group'}}
            if digest(candidate) not in candidates:
                raise CampaignFault('row_not_in_host_export')
            meta = row.get('meta', {})
            task = lineage.session_task(exported['identity']['epoch'], row['session'])
            if task['root'] != row.get('split_group') or task['id'] != meta.get('task_id'):
                raise CampaignFault('row_lineage_mismatch')
            source = row.get('source', {})
            verdict = episodes.admit(exported['identity']['epoch'], row['session'], source.get('store_sha256'))
            if (source.get('capture_sha256') not in permitted_hashes or source.get('store_sha256') not in permitted_hashes
                    or row.get('request_capture', {}).get('host_revision') != host_commit):
                raise CampaignFault('row_source_not_in_host_export')
            raw_path = next((Path(name) for name, checksum in exported['inputs'].items() if checksum == source['capture_sha256']), None)
            if raw_path is None or file_sha256(raw_path) != source['capture_sha256']:
                raise CampaignFault('host_capture_changed')
            capture = json.loads(raw_path.read_text())
            if capture.get('session_id') != row.get('session') or capture.get('epoch') != exported['identity']['epoch']:
                raise CampaignFault('capture_run_binding_mismatch')
            root = lineage.task(task['root'])
            specification = json.loads(root['spec'])
            sources[root['id']] = {'source_class': root['source_class'], 'training_allowed': True, 'parents': [],
                                   'permission_sha256': specification.get('permission_sha256')}
            sources[source['capture_sha256']] = {'source_class': 'factory_capture', 'training_allowed': True, 'parents': [root['id']]}
            approved[digest(row)] = {'root': task['root'], 'task_id': task['id'], 'partition': task['partition'],
                                    'sources': [source['capture_sha256']], 'episode': verdict}
    authority = {'run_id': exported['identity']['epoch'], 'host_commit': host_commit, 'lineage_sha256': lineage.freeze()}
    request = journal.directory / 'source-request.json'
    atomic_json(request, {'authority': authority, 'sources': sources, 'approved': approved, 'files': list(map(str, files))})
    destination = Path(files[0]).parent / 'source.json'
    if any(Path(path).parent != destination.parent for path in files):
        raise CampaignFault('dataset_splits_must_share_directory')
    run([sys.executable, str(Path(checkout) / 'scripts/bialy/custom_model/admission.py'), '--request', str(request),
         '--key', str(authority_key(journal.directory)), '--out', str(destination)], check=True)
    return {'source_manifest': str(destination), 'sha256': file_sha256(destination)}
