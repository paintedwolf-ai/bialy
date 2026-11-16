"""Export each student's training rows: the host's candidates, their admission, and one
qualified reviewer's verdict on every eligible row."""
import json
import subprocess
import sys
import uuid
from pathlib import Path

from .. import training_rows
from ..custom_model_run import student_binding
from ..host_bundle import Episode
from ..provenance import file_sha256
from .execution import atomic_json
from .journal import CampaignFault, canonical

ROWS = 'pw-training-rows/1'


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verified_candidates(manifest, host_commit):
    """The candidates a host export wrote, refused if its bundles or file changed since."""
    exported = json.loads(Path(manifest).read_text())
    if exported.get('schema') != 'pw-candidate-export/1' or exported['identity'].get('host_commit') != host_commit:
        raise CampaignFault('host_export_identity_mismatch')
    if any(file_sha256(Path(name) / 'manifest.json') != checksum for name, checksum in exported['bundles'].items()):
        raise CampaignFault('host_export_input_changed')
    path = Path(manifest).parent / 'candidates.jsonl'
    if file_sha256(path) != exported['candidates_sha256']:
        raise CampaignFault('host_export_rows_changed')
    return exported, lines(path)


class Exports:
    def __init__(self, reviewers, checkout, identity, catalog_sha256, process=subprocess.run):
        self.reviewers, self.checkout, self.identity = reviewers, Path(checkout), identity
        self.catalog_sha256, self.process = catalog_sha256, process

    def candidates(self, bundle, directory, binding):
        pointer = directory / 'candidates.json'
        expected = {'bundle': str(bundle), 'binding': binding, 'catalog_sha256': self.catalog_sha256}
        if pointer.exists():
            record = json.loads(pointer.read_text())
            if record['inputs'] != expected:
                raise CampaignFault('episode_export_inputs_changed')
            manifest = Path(record['manifest'])
        else:
            destination = directory / uuid.uuid4().hex
            binding_path = directory / 'binding.json'
            atomic_json(binding_path, binding)
            self.process([sys.executable, str(self.checkout / 'scripts/bialy/custom_model/collect.py'),
                          '--bundle', str(bundle), '--binding', str(binding_path), '--catalog-sha256', self.catalog_sha256,
                          '--out', str(destination)], check=True)
            manifest = destination / 'manifest.json'
            verified_candidates(manifest, binding['host_commit'])
            atomic_json(pointer, {'inputs': expected, 'manifest': str(manifest)})
        exported, rows = verified_candidates(manifest, binding['host_commit'])
        return manifest, exported, rows

    def __call__(self, inputs, output):
        directory = Path(output['directory']) / 'exports'
        directory.mkdir(exist_ok=True)
        episode = Episode(output['bundle'])
        results = {}
        for student, serving in self.identity['students'].items():
            target = directory / student
            target.mkdir(exist_ok=True)
            binding = student_binding(self.identity, serving)
            manifest, exported, candidates = self.candidates(output['bundle'], target, binding)
            records = training_rows.admit(candidates, episode, inputs['task'], output['receipt'])
            rows, reviewers = [], set()
            for record in records:
                if not training_rows.is_eligible(record):
                    continue
                row = {**record, 'meta': {'model': inputs['actor']['id']}}
                quality = self.reviewers.judge(row)
                reviewers.add((quality['reviewer_slot'], quality['judge_id'], quality['judge_revision'], quality['controls_sha256']))
                if quality['verdict'] == 'pass':
                    rows.append({**row, 'quality': quality})
            admissions, accepted = target / 'admissions.jsonl', target / 'rows.jsonl'
            for path, values in ((admissions, records), (accepted, rows)):
                body = ''.join(canonical(value) + '\n' for value in values)
                if path.exists() and path.read_text() != body:
                    raise CampaignFault('training_rows_changed')
                path.write_text(body)
            counts = {}
            for record in records:
                key = record['admission'].get('category') or 'excluded:' + record['admission']['reason']
                counts[key] = counts.get(key, 0) + 1
            rows_manifest = target / 'rows-manifest.json'
            atomic_json(rows_manifest, {'schema': ROWS, 'identity': binding, 'candidates_manifest_sha256': file_sha256(manifest),
                                        'bundles': exported['bundles'], 'admissions_sha256': file_sha256(admissions),
                                        'rows_sha256': file_sha256(accepted), 'counts': counts, 'reviewed_rows': len(rows),
                                        'reviewers': [dict(zip(('reviewer_slot', 'judge_id', 'judge_revision', 'controls_sha256'), item, strict=True))
                                                      for item in sorted(reviewers)]})
            results[student] = {'manifest': str(rows_manifest), 'rows': len(rows)}
        return results
