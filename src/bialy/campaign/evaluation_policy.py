"""evaluation-policy.json: the behavioral thresholds and comparison design, frozen before
any student result exists. No student result is read without it."""
import json
from pathlib import Path

import yaml

from .execution import atomic_json
from .journal import CampaignFault, digest

CONFIG = Path(__file__).resolve().parents[3] / 'config/campaign/evaluation-policy.yaml'
SCHEMA = 'pw-evaluation-policy/1'
FILE = 'evaluation-policy.json'


def load(path=CONFIG):
    value = yaml.safe_load(Path(path).read_text())
    design = value['design']
    if (value['decides'] != 'system_test' or value['diagnostic'] != 'development'
            or type(design['attempts_per_task']) is not int or design['attempts_per_task'] < 2
            or type(design['bootstrap_samples']) is not int or design['bootstrap_samples'] < 1000
            or not 0 < design['interval'] < 1):
        raise CampaignFault('evaluation_policy_invalid')
    return value


def freeze(directory, policy):
    """Write the policy once; a later freeze must be the same policy."""
    path = Path(directory) / FILE
    body = {'schema': SCHEMA, 'policy': policy, 'policy_sha256': digest(policy)}
    if path.exists():
        if json.loads(path.read_text()) != body:
            raise CampaignFault('evaluation_policy_changed')
    else:
        atomic_json(path, body)
    return body['policy_sha256']


def frozen(directory, plan):
    """The policy the plan froze; refused while it is absent or differs."""
    path = Path(directory) / FILE
    if not path.exists():
        raise CampaignFault('evaluation_policy_unfrozen')
    body = json.loads(path.read_text())
    if body.get('schema') != SCHEMA or digest(body['policy']) != body['policy_sha256'] or body['policy_sha256'] != plan.get('policy_sha256'):
        raise CampaignFault('evaluation_policy_unfrozen')
    return body['policy']
