"""A campaign configuration: the pinned sources, rosters, storage and serving a plan freezes.

Validation is offline: it reads only the file and the factory's own pinned specifications.
"""
import re
from pathlib import Path

import yaml

from ..student_host.specification import load as students
from .journal import CampaignFault

REQUIRED = {'profile', 'epoch', 'checkout', 'host_commit', 'host_image', 'factory_root', 'corpus', 'models', 'teachers',
            'reviewers', 'server_image', 'gateway', 'llama', 'llama_revision', 'full_training', 'device', 'mirror'}
OPTIONAL = {'students', 'human_review', 'pilot_training', 'remote', 'head_rows', 'selector_policy'}
COMMIT = re.compile(r'[0-9a-f]{40}')
EPOCH = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,49}')


def load(path, profile):
    """The configuration for `profile`, with its student list resolved."""
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or REQUIRED - set(value) or set(value) - REQUIRED - OPTIONAL:
        raise CampaignFault('campaign_configuration_keys', details={
            'missing': sorted(REQUIRED - set(value or {})), 'unknown': sorted(set(value or {}) - REQUIRED - OPTIONAL)})
    pinned = students()['models']
    roster = value.get('students') or profile.get('students') or list(pinned)
    problems = []
    if value['profile'] != profile['name']:
        problems.append('profile')
    if not COMMIT.fullmatch(str(value['host_commit'])) or not COMMIT.fullmatch(str(value['llama_revision'])):
        problems.append('commits')
    if not EPOCH.fullmatch(str(value['epoch'])):
        problems.append('epoch')
    if len(set(roster)) != len(roster) or set(roster) - set(pinned):
        problems.append('students')
    if not value['teachers'] or not value['reviewers'] or len(set(value['reviewers'])) != len(value['reviewers']):
        problems.append('rosters')
    if type(value.get('human_review', False)) is not bool or value['full_training'] not in ('local', 'remote'):
        problems.append('modes')
    if value['full_training'] == 'remote' and not value.get('remote'):
        problems.append('remote')
    if ('selector' in profile['segments']) != bool(value.get('head_rows')):
        problems.append('head_rows')
    if value.get('selector_policy') and profile['selector'] != 'supplied_or_engine_disabled':
        problems.append('selector_policy')
    if problems:
        raise CampaignFault('campaign_configuration_invalid', details={'fields': problems})
    return {**value, 'students': roster, 'human_review': value.get('human_review', False)}
