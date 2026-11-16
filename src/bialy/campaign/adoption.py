"""The host adoption boundary: the student segment runs only on a host commit whose
decision release and thresholds are P0's."""
import json
import subprocess

import yaml

from . import selector
from .journal import CampaignFault

RELEASE = 'lycaon/config/packs/painted-wolf/platform/host/decision-release.json'


def show(checkout, commit, path, process=subprocess.run):
    result = process(['git', '-C', str(checkout), 'show', '%s:%s' % (commit, path)], capture_output=True, text=True)
    if result.returncode:
        raise CampaignFault('adopted_commit_unreadable', details={'commit': commit, 'path': path})
    return result.stdout


def differences(policy, release, decisions):
    """Every pin and threshold of the adopted commit that is not P0's."""
    out = []
    for head, pinned in policy['heads'].items():
        found = (release.get('heads') or {}).get(head, {}).get('sha256')
        if found != pinned['sha256']:
            out.append({'head': head, 'expected': pinned['sha256'], 'found': found})
    for name, value in policy['thresholds'].items():
        found = selector.measure(decisions, name)
        if found != value:
            out.append({'threshold': name, 'expected': value, 'found': found})
    if decisions.get('version') != policy['decision_contract']:
        out.append({'decision_contract': policy['decision_contract'], 'found': decisions.get('version')})
    return out


def check(checkout, commit, policy, process=subprocess.run):
    """The adopted commit's facts, refused unless its decision configuration equals P0."""
    if not selector.COMMIT.fullmatch(str(commit)):
        raise CampaignFault('adopted_commit_invalid')
    release = json.loads(show(checkout, commit, RELEASE, process))
    decisions = yaml.safe_load(show(checkout, commit, selector.load()['decisions'], process))
    found = differences(policy, release, decisions)
    if found:
        raise CampaignFault('adopted_commit_differs_from_p0', details={'commit': commit, 'differences': found})
    return {'host_commit': commit, 'policy': policy['identity'], 'release': release.get('release')}
