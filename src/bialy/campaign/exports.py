"""Export each student's host receipts and retain response reviews across restarts."""
import json
import subprocess
import sys
import uuid
from pathlib import Path

from .. import custom_quality
from ..custom_model_run import student_binding
from ..dpo import training_eligible
from ..provenance import file_sha256
from .execution import atomic_json
from .journal import CampaignFault, canonical, digest
from .sources import verified_exports


class Exports:
    def __init__(self, factory, checkout, identity, catalog_sha256, process=subprocess.run):
        self.factory, self.checkout, self.identity = factory, Path(checkout), identity
        self.catalog_sha256, self.process = catalog_sha256, process

    def collect(self, fleet, directory, binding, quality=None):
        pointer = directory / ('reviewed.json' if quality else 'candidates.json')
        expected = {'fleet': str(fleet), 'binding': binding, 'catalog_sha256': self.catalog_sha256,
                    'quality_sha256': file_sha256(quality) if quality else None}
        if pointer.exists():
            record = json.loads(pointer.read_text())
            if record['inputs'] != expected:
                raise CampaignFault('episode_export_inputs_changed')
            manifest = Path(record['manifest'])
        else:
            destination = directory / uuid.uuid4().hex
            binding_path = directory / 'binding.json'
            atomic_json(binding_path, binding)
            arguments = [sys.executable, str(self.checkout / 'scripts/bialy/custom_model/collect.py'),
                         '--fleet', str(fleet), '--binding', str(binding_path), '--catalog-sha256', self.catalog_sha256,
                         '--out', str(destination)]
            if quality:
                arguments += ['--quality', str(quality)]
            self.process(arguments, check=True)
            manifest = destination / 'manifest.json'
            verified_exports([manifest], binding['host_commit'])
            atomic_json(pointer, {'inputs': expected, 'manifest': str(manifest)})
        _, _, rows = verified_exports([manifest], binding['host_commit'])
        return manifest, rows

    def __call__(self, inputs, output):
        directory = Path(output['directory']) / 'exports'
        directory.mkdir(exist_ok=True)
        facts = json.loads((Path(output['directory']) / 'host-evidence.json').read_text())
        transcript = directory / 'transcript.jsonl'
        transcript_body = ''.join(canonical({**session, 'session_id': session['id']}) + '\n' for session in facts['sessions'])
        if transcript.exists() and transcript.read_text() != transcript_body:
            raise CampaignFault('episode_review_transcript_changed')
        transcript.write_text(transcript_body)
        results = {}
        for student, serving in self.identity['students'].items():
            target = directory / student
            target.mkdir(exist_ok=True)
            binding = student_binding(self.identity, serving)
            _, candidates = self.collect(output['fleet'], target, binding)
            reviews = []
            for key, row in candidates.items():
                review_directory = target / ('review-' + key)
                review_directory.mkdir(exist_ok=True)
                reviewed = review_directory / 'review.json'
                annotated = {**row, 'meta': {'model': inputs['actor']['id']}}
                validation = row['validation']
                schema = custom_quality.review_schema([validation['receipt_id'], *validation.get('invocation_receipt_ids', [])])
                expected = digest({'row': annotated, 'transcript': transcript_body, 'system': custom_quality.SYSTEM, 'schema': schema, 'seed': 7, 'reviewer': {'id': self.factory.reviewer().id, 'revision': self.factory.reviewer().revision}})
                if reviewed.exists():
                    record = json.loads(reviewed.read_text())
                    if record['inputs_sha256'] != expected:
                        raise CampaignFault('response_review_inputs_changed')
                    reports = record['reports']
                else:
                    intent_path = review_directory / 'attempt.json'
                    if intent_path.exists():
                        intent = json.loads(intent_path.read_text())
                        if intent['inputs_sha256'] != expected:
                            raise CampaignFault('response_review_inputs_changed')
                        destination = Path(intent['output'])
                        try:
                            reports = [json.loads(line) for line in destination.read_text().splitlines() if line.strip()]
                        except (OSError, json.JSONDecodeError) as exc:
                            raise CampaignFault('response_review_outcome_unknown') from exc
                        if len(reports) != 1:
                            raise CampaignFault('response_review_outcome_unknown')
                    else:
                        rows = review_directory / 'row.jsonl'
                        rows.write_text(canonical(annotated) + '\n')
                        destination = review_directory / (uuid.uuid4().hex + '.jsonl')
                        atomic_json(intent_path, {'inputs_sha256': expected, 'output': str(destination)})
                        custom_quality.judge_rows(self.factory, rows, transcript, destination)
                        reports = [json.loads(line) for line in destination.read_text().splitlines() if line.strip()]
                    atomic_json(reviewed, {'inputs_sha256': expected, 'reports': reports})
                reviews.extend(reports)
            quality = target / 'quality.jsonl'
            body = ''.join(canonical(review) + '\n' for review in reviews)
            if quality.exists() and quality.read_text() != body:
                raise CampaignFault('response_reviews_changed')
            quality.write_text(body)
            manifest, candidates = self.collect(output['fleet'], target, binding, quality)
            results[student] = {'manifest': str(manifest), 'rows': sum(training_eligible(row) for row in candidates.values())}
        return results
