"""Export each student's training rows: the host's candidates, their admission, and one
review of every eligible row, retained across restarts."""
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
from .journal import CampaignFault, canonical, digest
from .review_contract import ACTION, judge_rows

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
    def __init__(self, factory, checkout, identity, catalog_sha256, process=subprocess.run):
        self.factory, self.checkout, self.identity = factory, Path(checkout), identity
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

    def review(self, row, directory, transcript, transcript_body):
        """The reviewer's report on one eligible row; a review interrupted mid-call is never repeated silently."""
        key = digest({'request': row['request_capture']['sha256'], 'response': row['response_sha256']})[:32]
        review_directory = directory / ('review-' + key)
        review_directory.mkdir(exist_ok=True)
        reviewed = review_directory / 'review.json'
        execution = row['execution']
        schema = ACTION.bound([execution['receipt_id'], *(i for call in execution['calls'] for i in call.get('operation_ids', []))])
        reviewer = self.factory.reviewer()
        expected = digest({'row': row, 'transcript': transcript_body, 'system': ACTION.system, 'schema': schema, 'seed': 7,
                           'reviewer': {'id': reviewer.id, 'revision': reviewer.revision}})
        if reviewed.exists():
            record = json.loads(reviewed.read_text())
            if record['inputs_sha256'] != expected:
                raise CampaignFault('response_review_inputs_changed')
            return record['reports']
        intent_path = review_directory / 'attempt.json'
        if intent_path.exists():
            intent = json.loads(intent_path.read_text())
            if intent['inputs_sha256'] != expected:
                raise CampaignFault('response_review_inputs_changed')
            try:
                reports = lines(intent['output'])
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignFault('response_review_outcome_unknown') from exc
            if len(reports) != 1:
                raise CampaignFault('response_review_outcome_unknown')
        else:
            source = review_directory / 'row.jsonl'
            source.write_text(canonical(row) + '\n')
            destination = review_directory / (uuid.uuid4().hex + '.jsonl')
            atomic_json(intent_path, {'inputs_sha256': expected, 'output': str(destination)})
            judge_rows(self.factory, source, transcript, destination)
            reports = lines(destination)
        atomic_json(reviewed, {'inputs_sha256': expected, 'reports': reports})
        return reports

    def __call__(self, inputs, output):
        directory = Path(output['directory']) / 'exports'
        directory.mkdir(exist_ok=True)
        episode = Episode(output['bundle'])
        transcript = directory / 'transcript.jsonl'
        transcript_body = ''.join(canonical({**session, 'session_id': session['id'], 'messages': episode.messages(session)}) + '\n'
                                  for session in episode.sessions)
        if transcript.exists() and transcript.read_text() != transcript_body:
            raise CampaignFault('episode_review_transcript_changed')
        transcript.write_text(transcript_body)
        results = {}
        for student, serving in self.identity['students'].items():
            target = directory / student
            target.mkdir(exist_ok=True)
            binding = student_binding(self.identity, serving)
            manifest, exported, candidates = self.candidates(output['bundle'], target, binding)
            records = training_rows.admit(candidates, episode, inputs['task'], output['receipt'])
            rows = []
            for record in records:
                if not training_rows.is_eligible(record):
                    continue
                row = {**record, 'meta': {'model': inputs['actor']['id']}}
                reports = self.review(row, target, transcript, transcript_body)
                if reports and reports[0].get('verdict') == 'pass':
                    rows.append({**row, 'quality': reports[0]})
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
                                        'rows_sha256': file_sha256(accepted), 'counts': counts, 'reviewed_rows': len(rows)})
            results[student] = {'manifest': str(rows_manifest), 'rows': len(rows)}
        return results
