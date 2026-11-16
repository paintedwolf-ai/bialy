"""Campaign profiles: which segments run and the workload each stage is given."""
from pathlib import Path

import yaml

from .journal import CampaignFault

CONFIG = Path(__file__).resolve().parents[3] / 'config/campaign.yaml'
SEGMENTS = ('selector', 'student')
SELECTORS = ('engine_disabled', 'supplied_or_engine_disabled', 'p0')
RECIPES = ('dry_run', 'pilot', 'full')
PARTITIONS = ('train', 'validation', 'system_test')


def load(name, path=CONFIG):
    profiles = yaml.safe_load(Path(path).read_text())['profiles']
    if name not in profiles:
        raise CampaignFault('campaign_profile_unknown', details={'profile': name})
    profile = dict(profiles[name], name=name)
    segments, collection = profile['segments'], profile['collection']
    if (not segments or any(segment not in SEGMENTS for segment in segments) or 'student' not in segments
            or profile['selector'] not in SELECTORS or (profile['selector'] == 'p0') != ('selector' in segments)
            or set(collection) != set(PARTITIONS) or any(type(count) is not int or count < 1 for count in collection.values())
            or type(profile['judged']) is not bool or profile['recipe'] not in RECIPES or profile['quota'] not in ('pilot', 'full')):
        raise CampaignFault('campaign_profile_invalid', details={'profile': name})
    rows = profile.get('rows')
    if rows is not None and (set(rows) != set(PARTITIONS) or type(profile.get('max_example_tokens')) is not int):
        raise CampaignFault('campaign_profile_invalid', details={'profile': name})
    return profile
