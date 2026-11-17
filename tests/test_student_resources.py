import json

import pytest

from bialy.student_host.resources import SETTLE_SECONDS, ResourceError, Resources


class Provider:
    """A provider reduced to the ledger's protocol, with each answer under the test's control."""

    def __init__(self):
        self.instances, self.created, self.acknowledges, self.fail_create = {}, 0, True, None

    def owned(self, token):
        return [dict(item) for item in self.instances.values() if item['token'] == token]

    def create(self, token, specification, startup, avoid=()):
        self.created += 1
        instance = {'id': 'instance-%d' % self.created, 'token': token, 'state': 'pending', 'endpoint': None,
                    'interruptible': False, 'children': [], 'host': 'machine-%d' % self.created}
        self.instances[instance['id']] = instance
        if self.fail_create:
            raise self.fail_create
        return dict(instance)

    def remaining(self, instance):
        return [instance['id']] if instance['id'] in self.instances else []

    def destroy(self, instance):
        return self.instances.pop(instance['id'], None) is not None and self.acknowledges


def ledger(tmp_path, provider, now=100):
    resources = Resources(tmp_path, provider, clock=lambda: now)
    resources.reap()
    return resources


def at(tmp_path, provider, now):
    return Resources(tmp_path, provider, clock=lambda: now)


def test_a_lost_create_response_is_adopted_by_token_without_a_second_instance(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    provider.fail_create = ConnectionError('response lost')
    with pytest.raises(ConnectionError):
        resources.allocate(token, 'startup')
    provider.fail_create = None
    assert resources.allocate(token, 'startup')['instance'] == 'instance-1'
    assert provider.created == 1
    assert resources.release(token)['state'] == 'released' and not provider.instances


def test_an_unknown_create_never_creates_again_and_stays_doubted_until_the_listing_can_be_trusted(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    provider.fail_create = ConnectionError('response lost')
    with pytest.raises(ConnectionError):
        resources.allocate(token, 'startup')
    hidden = provider.instances.popitem()[1]
    with pytest.raises(ResourceError, match='unknown'):
        resources.allocate(token, 'startup')
    with pytest.raises(ResourceError, match='not acknowledged'):
        resources.release(token)
    assert provider.created == 1
    provider.instances[hidden['id']] = hidden
    assert at(tmp_path, provider, 130).reap()[0]['state'] == 'released'
    assert not provider.instances


def test_an_unknown_create_that_made_nothing_settles_once_the_doubt_runs_out(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    provider.create = lambda *args: (_ for _ in ()).throw(ConnectionError('request lost'))
    with pytest.raises(ConnectionError):
        resources.allocate(token, 'startup')
    assert at(tmp_path, provider, 99 + SETTLE_SECONDS).reap()[0]['state'] == 'cleanup_unconfirmed'
    assert at(tmp_path, provider, 100 + SETTLE_SECONDS).reap()[0]['state'] == 'released'


def test_an_unacknowledged_destroy_of_a_new_instance_is_repeated_until_acknowledged(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    resources.allocate(token, 'startup')
    unlisted = provider.instances.popitem()[1]
    with pytest.raises(ResourceError, match='not acknowledged'):
        resources.release(token)
    provider.instances[unlisted['id']] = unlisted
    assert at(tmp_path, provider, 110).reap()[0]['state'] == 'released'
    assert not provider.instances


def test_a_released_lease_is_watched_for_a_late_instance_and_then_left_alone(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    late = dict(resources.allocate(token, 'startup')['instances'][0], token=token)
    resources.release(token)
    provider.instances[late['id']] = late
    assert at(tmp_path, provider, 150).reap()[0]['state'] == 'released'
    assert not provider.instances
    provider.owned = lambda token: pytest.fail('a settled lease was checked again')
    assert at(tmp_path, provider, 150 + SETTLE_SECONDS).reap() == []


def test_the_reaper_releases_expired_leases_and_leaves_live_and_foreign_ones(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    expiring, kept = (resources.intent({}, 120, 3600)['token'] for _ in range(2))
    resources.allocate(expiring, 'startup')
    resources.allocate(kept, 'startup')
    provider.instances['foreign'] = {'id': 'foreign', 'token': 'someone-else'}
    at(tmp_path, provider, 200).renew(kept, 120)
    results = at(tmp_path, provider, 221).reap()
    assert [(lease['token'], lease['state']) for lease in results] == [(expiring, 'released')]
    assert sorted(provider.instances) == ['foreign', 'instance-2']


def test_a_renewal_delayed_past_the_expiry_keeps_a_lease_the_reaper_has_not_taken(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    resources.allocate(token, 'startup')
    at(tmp_path, provider, 500).renew(token, 120)
    assert at(tmp_path, provider, 501).reap() == [] and provider.instances
    at(tmp_path, provider, 700).reap()
    with pytest.raises(ResourceError, match='cannot renew'):
        at(tmp_path, provider, 701).renew(token, 120)


def test_no_renewal_carries_a_lease_past_its_deadline(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 600)['token']
    resources.allocate(token, 'startup')
    at(tmp_path, provider, 650).renew(token, 120)
    assert json.loads((tmp_path / (token + '.json')).read_text())['expires_at'] == 700
    with pytest.raises(ResourceError, match='deadline'):
        at(tmp_path, provider, 700).renew(token, 120)
    assert at(tmp_path, provider, 700).reap()[0]['state'] == 'released' and not provider.instances


def test_an_unhealthy_reaper_refuses_new_leases_and_a_failed_release_is_never_claimed(tmp_path):
    provider = Provider()
    with pytest.raises(ResourceError, match='reaper'):
        Resources(tmp_path, provider, clock=lambda: 100).intent({}, 120, 3600)
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    resources.allocate(token, 'startup')
    listing = provider.owned
    provider.owned = lambda token: (_ for _ in ()).throw(ConnectionError('offline'))
    with pytest.raises(ConnectionError):
        resources.release(token)
    assert json.loads((tmp_path / (token + '.json')).read_text())['state'] == 'cleanup_unconfirmed'
    assert resources.reap()[0]['state'] == 'cleanup_unconfirmed'
    provider.owned = listing
    assert resources.reap()[0]['state'] == 'released'


def test_recovery_requires_the_same_live_instance(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    resources.allocate(token, 'startup')
    assert resources.recover(token, 180)['expires_at'] == 280
    provider.instances.clear()
    with pytest.raises(ResourceError, match='missing'):
        resources.recover(token, 180)
    with pytest.raises(ResourceError, match='cannot be recovered'):
        at(tmp_path, provider, 281).recover(token, 180)


def test_children_a_later_snapshot_omits_are_still_destroyed(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    resources.allocate(token, 'startup')
    provider.instances['instance-1']['children'] = [['volume', 'disk'], ['ip', 'address']]
    resources.observe(token)
    provider.instances['instance-1']['children'] = []
    destroyed = []
    destroy = provider.destroy
    provider.destroy = lambda instance: (destroyed.append(instance['children']), destroy(instance))[1]
    assert resources.release(token)['state'] == 'released'
    assert destroyed == [[['ip', 'address'], ['volume', 'disk']]]


def test_an_unreadable_lease_is_released_by_its_token_and_does_not_block_the_others(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    damaged, expiring = (resources.intent({}, 120, 3600)['token'] for _ in range(2))
    resources.allocate(damaged, 'startup')
    resources.allocate(expiring, 'startup')
    (tmp_path / (damaged + '.json')).write_text('{not json')
    states = {lease['token']: lease['state'] for lease in at(tmp_path, provider, 300).reap()}
    assert states == {damaged: 'released', expiring: 'released'} and not provider.instances


def test_an_unreadable_lease_with_nothing_listed_is_doubted_before_it_settles(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    damaged = resources.intent({}, 120, 3600)['token']
    (tmp_path / (damaged + '.json')).write_text('')
    assert at(tmp_path, provider, 300).reap()[0]['state'] == 'cleanup_unconfirmed'
    assert at(tmp_path, provider, 300 + SETTLE_SECONDS).reap()[0]['state'] == 'released'


def test_two_instances_under_one_token_are_refused_and_both_destroyed(tmp_path):
    provider = Provider()
    resources = ledger(tmp_path, provider)
    token = resources.intent({}, 120, 3600)['token']
    for name in ('first', 'second'):
        provider.instances[name] = {'id': name, 'token': token, 'children': []}
    with pytest.raises(ResourceError, match='multiple'):
        resources.allocate(token, 'startup')
    assert provider.created == 0
    assert resources.release(token)['state'] == 'released' and not provider.instances


def test_an_avoided_machine_is_passed_to_the_provider_until_the_avoidance_lapses(tmp_path):
    from bialy.student_host.resources import AVOID_SECONDS
    provider, seen = Provider(), []
    create = provider.create
    provider.create = lambda token, specification, startup, avoid=(): (seen.append(set(avoid)), create(token, specification, startup))[1]
    resources = ledger(tmp_path, provider)
    first = resources.intent({}, 120, 3600)['token']
    resources.allocate(first, 'startup')
    resources.avoid(first)
    resources.release(first)
    resources.allocate(resources.intent({}, 120, 3600)['token'], 'startup')
    later = at(tmp_path, provider, 100 + AVOID_SECONDS)
    later.reap()
    later.allocate(later.intent({}, 120, 3600)['token'], 'startup')
    assert seen == [set(), {'machine-1'}, set()]
