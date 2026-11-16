"""Qualify each reviewer slot for each reviewing role on versioned controls.

A role's controls are positives and false successes shaped as that role's own input. A
slot fills a role only after it passes every positive control of the role and accepts no
negative one. A role with no qualified slot stops the campaign as `reviewers_unqualified`.
"""
from pathlib import Path

import yaml

from ..training_rows import SCHEMA, turn_outcome
from .journal import CampaignFault, digest
from .review_contract import ROLES, action_payload, episode_payload

CONTROLS = Path(__file__).resolve().parents[3] / 'config/campaign/reviewer-controls.yaml'


def load(path=CONTROLS):
    """Every role's controls, each role identified by the digest of its own section."""
    value = yaml.safe_load(Path(path).read_text())
    roles = {}
    for role in ROLES:
        section = value.get(role) or {}
        controls = section.get('controls') or []
        ids = [item['id'] for item in controls]
        negatives = {item.get('negative') for item in controls if item['kind'] == 'negative'}
        required = set(section.get('required_negatives') or [])
        if (len(ids) != len(set(ids)) or not required or required - negatives
                or not any(item['kind'] == 'positive' for item in controls)
                or any(item['kind'] not in {'positive', 'negative'} for item in controls)):
            raise CampaignFault('reviewer_controls_incomplete', details={'role': role, 'missing_negatives': sorted(required - negatives)})
        roles[role] = {'version': value['version'], 'controls': controls,
                       'sha256': digest({'version': value['version'], 'role': role, 'section': section})}
    return roles


def episode_control(control):
    """A control as the episode rubric sees an episode: its receipt and its transcript."""
    receipt = {**control['receipt'], 'task': control['task'], 'operations': control['operations']}
    return episode_payload(receipt, [{'id': 'root', 'messages': control['transcript']}])


def action_control(control):
    """A control as the action rubric sees a training row."""
    calls = control['calls']
    response = {'role': 'assistant', 'text': control['response'].get('text', ''), 'tool_calls': control['response'].get('tool_calls', [])}
    tools = [{'type': 'function', 'function': {'name': name}} for name in control['tools']]
    return action_payload({
        'schema': SCHEMA, 'request_capture': {'messages': control['request'], 'tools': tools}, 'target': response,
        'execution': {'receipt_id': 'out-' + control['id'], 'calls': calls, 'committed': not calls},
        'admission': {'eligibility': 'eligible', 'category': control['category'], 'semantic_outcome': turn_outcome(calls)}})


PAYLOADS = {'episode': episode_control, 'action': action_control}


def qualify(reviewers, controls=None):
    """Qualify every slot for every role; returns each role's results by slot and raises
    when a role has no qualified slot."""
    controls = controls or load()
    results = {}
    for role, definition in ROLES.items():
        chosen, results[role] = controls[role], {}
        for slot, model in enumerate(reviewers.models):
            prior = reviewers.qualification(role, slot, model)
            if prior and prior['controls_sha256'] == chosen['sha256']:
                results[role][str(slot)] = prior
                continue
            attempts = []
            for control in chosen['controls']:
                try:
                    answer = reviewers.ask(role, slot, model, PAYLOADS[role](control), 'control:' + chosen['sha256'])
                    correct = definition.accepts(answer) is (control['kind'] == 'positive')
                    attempts.append({'id': control['id'], 'kind': control['kind'], 'correct': correct, 'answer': answer})
                except CampaignFault as exc:
                    attempts.append({'id': control['id'], 'kind': control['kind'], 'correct': False, 'fault': exc.record()})
            accepted_negatives = [item['id'] for item in attempts if item['kind'] == 'negative' and not item['correct']]
            result = {'slot': str(slot), 'role': role, 'model': model.id, 'controls_version': chosen['version'],
                      'controls_sha256': chosen['sha256'], 'attempts': attempts, 'accepted_negatives': accepted_negatives,
                      'qualified': all(item['correct'] for item in attempts)}
            reviewers.record(role, slot, model, result)
            results[role][str(slot)] = result
    unqualified = sorted(role for role, slots in results.items() if not any(result['qualified'] for result in slots.values()))
    if unqualified:
        raise CampaignFault('reviewers_unqualified', details={'roles': unqualified, 'results': results})
    return results
