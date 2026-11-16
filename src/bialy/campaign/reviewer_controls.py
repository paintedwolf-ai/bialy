"""Qualify each reviewer slot on versioned positive and false-success controls.

A slot reviews episodes only after it passes every positive control and accepts no
negative one; a slot that passes any negative is disabled. With no slot left the
campaign stops as `reviewers_unqualified`.
"""
from pathlib import Path

import yaml

from .journal import CampaignFault, digest
from .review_contract import episode_payload

CONTROLS = Path(__file__).resolve().parents[3] / 'config/campaign/reviewer-controls.yaml'


def load(path=CONTROLS):
    value = yaml.safe_load(Path(path).read_text())
    controls = value['controls']
    ids = [item['id'] for item in controls]
    negatives = {item.get('negative') for item in controls if item['kind'] == 'negative'}
    missing = set(value['required_negatives']) - negatives
    if (len(ids) != len(set(ids)) or missing or not any(item['kind'] == 'positive' for item in controls)
            or any(item['kind'] not in {'positive', 'negative'} for item in controls)):
        raise CampaignFault('reviewer_controls_incomplete', details={'missing_negatives': sorted(missing)})
    return {'version': value['version'], 'controls': controls, 'sha256': digest(value)}


def payload(control):
    """A control as the episode rubric sees an episode: its receipt and its transcript."""
    receipt = {**control['receipt'], 'task': control['task'], 'operations': control['operations']}
    return episode_payload(receipt, [{'id': 'root', 'messages': control['transcript']}])


def accepted(answer):
    return all(value == 'pass' for value in answer['dimensions'].values())


def qualify(reviewers, controls=None):
    """Qualify every slot; returns each slot's result and raises when none qualifies."""
    controls = controls or load()
    results = {}
    for slot, model in enumerate(reviewers.models):
        prior = reviewers.qualification(slot, model)
        if prior and prior['controls_sha256'] == controls['sha256']:
            results[str(slot)] = prior
            continue
        attempts = []
        for control in controls['controls']:
            try:
                answer = reviewers.ask(slot, model, payload(control), 'control:' + controls['sha256'])
                correct = accepted(answer) if control['kind'] == 'positive' else not accepted(answer)
                attempts.append({'id': control['id'], 'kind': control['kind'], 'correct': correct, 'answer': answer})
            except CampaignFault as exc:
                attempts.append({'id': control['id'], 'kind': control['kind'], 'correct': False, 'fault': exc.record()})
        accepted_negatives = [item['id'] for item in attempts if item['kind'] == 'negative' and not item['correct']]
        result = {'slot': str(slot), 'model': model.id, 'controls_version': controls['version'],
                  'controls_sha256': controls['sha256'], 'attempts': attempts, 'accepted_negatives': accepted_negatives,
                  'qualified': all(item['correct'] for item in attempts)}
        reviewers.record(slot, model, result)
        results[str(slot)] = result
    if not any(result['qualified'] for result in results.values()):
        raise CampaignFault('reviewers_unqualified', details={'results': results})
    return results
