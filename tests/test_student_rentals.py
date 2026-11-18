"""Every point at which a rental can fail, walked through the real provider clients and the
ledger. Whatever fails, one instance at most is ever created and the reaper ends up with
nothing owned and nothing foreign touched."""
import json
import os
import time

import pytest
from rental_accounts import ACCOUNTS, FAULTS, Died, Machine, configuration

from bialy.student_host import machines, resources
from bialy.student_host.resources import SETTLE_SECONDS, Resources


@pytest.fixture(autouse=True)
def unsynced(monkeypatch):
    """These tests write thousands of ledger files and none of them survives a power cut."""
    monkeypatch.setattr(os, 'fsync', lambda descriptor: None)


# Listings a new instance is missing from: none, a few, and more than a controller makes before it gives up.
LAGS = [0, 3, 25]


def rent(root, account, monkeypatch, machine=Machine):
    """One rental from intent to release, as a remote job drives it. Returns what the controller saw."""
    monkeypatch.setattr(machines, 'provider', lambda provision: account.client())
    config = configuration(root, account.name)
    job = root / 'job'
    job.mkdir(parents=True)
    Resources(config['provisioning']['registry'], account.client()).reap()
    try:
        with machines.connect(config, job, machine, sleep=lambda _: None) as (remote, supervisor):
            supervisor.check()
            return 'connected', remote
    except (Exception, Died) as exc:
        return type(exc).__name__, exc


def reap_until_settled(root, account):
    """The independent reaper, running on while the lease expires and every doubt runs out."""
    account.heal()
    registry, start = root / 'registry', time.time()
    for offset in (*range(0, SETTLE_SECONDS + machines.LEASE_SECONDS + 60, 20), 2 * SETTLE_SECONDS + 500):
        Resources(registry, account.client(), clock=lambda offset=offset: start + offset).reap()
    return [json.loads(path.read_text()) for path in registry.glob('*.json') if path.name != 'reaper.json']


def assert_nothing_stranded(root, account, where, most=1):
    cleanup = root / 'job' / 'cleanup.json'
    if cleanup.exists() and json.loads(cleanup.read_text())['status'] == 'confirmed':
        assert account.owned() == [], 'release was confirmed while the account still held resources: %s' % where
    leases = reap_until_settled(root, account)
    assert account.created <= most, 'more instances were created than the job may rent: %s' % where
    assert account.peak <= 1, 'two instances existed at once: %s' % where
    assert account.owned() == [], 'an owned resource outlived the reaper: %s' % where
    assert all(lease['state'] == 'released' for lease in leases), 'a lease never settled: %s' % where
    untouched = ACCOUNTS[account.name]()
    assert set(vars(account).get('instances', {})) == set(vars(untouched).get('instances', {}))
    assert set(vars(account).get('servers', {})) == set(vars(untouched).get('servers', {}))
    assert vars(account).get('children') == vars(untouched).get('children')


@pytest.mark.parametrize('provider', sorted(ACCOUNTS))
@pytest.mark.parametrize('lag', LAGS)
def test_a_clean_rental_is_one_instance_released_and_confirmed(tmp_path, monkeypatch, provider, lag):
    account = ACCOUNTS[provider](lag=lag)
    outcome, remote = rent(tmp_path, account, monkeypatch)
    assert outcome == 'connected' and account.created == 1 and account.owned() == []
    assert json.loads((tmp_path / 'job' / 'cleanup.json').read_text())['status'] == 'confirmed'
    assert remote.settings['port'] == (41235 if provider == 'vastai' else 22)
    assert_nothing_stranded(tmp_path, account, 'clean')


@pytest.mark.parametrize('provider', sorted(ACCOUNTS))
@pytest.mark.parametrize('lag', LAGS)
@pytest.mark.parametrize('fault', FAULTS)
def test_no_provider_failure_point_strands_or_duplicates_an_instance(tmp_path, monkeypatch, provider, lag, fault):
    clean = ACCOUNTS[provider](lag=lag)
    rent(tmp_path / 'clean', clean, monkeypatch)
    assert len(clean.calls) > 5
    for index in range(len(clean.calls)):
        root = tmp_path / str(index)
        account = ACCOUNTS[provider](lag=lag)
        account.fail(index, fault)
        rent(root, account, monkeypatch)
        assert_nothing_stranded(root, account, '%s at provider call %d %r' % (fault, index, clean.calls[index][2:8]))


@pytest.mark.parametrize('provider', sorted(ACCOUNTS))
@pytest.mark.parametrize('written', [False, True])
def test_no_ledger_write_can_be_the_controllers_last_and_strand_an_instance(tmp_path, monkeypatch, provider, written):
    real = resources.write
    counted = []

    def counting(path, value):
        counted.extend([path] * (path.name != 'reaper.json'))
        real(path, value)
    monkeypatch.setattr(resources, 'write', counting)
    rent(tmp_path / 'clean', ACCOUNTS[provider](), monkeypatch)
    assert len(counted) > 5
    for index in range(len(counted)):
        root, account, seen = tmp_path / str(index), ACCOUNTS[provider](), []

        def dying(path, value, index=index, seen=seen):
            if path.name == 'reaper.json':
                return real(path, value)
            seen.append(path)
            if len(seen) - 1 > index or (len(seen) - 1 == index and not written):
                raise Died()
            real(path, value)
            if len(seen) - 1 == index:
                raise Died()
        monkeypatch.setattr(resources, 'write', dying)
        rent(root, account, monkeypatch)
        monkeypatch.setattr(resources, 'write', real)
        assert_nothing_stranded(root, account, 'death at ledger write %d (%s)' % (index, 'after' if written else 'before'))


@pytest.mark.parametrize('provider', sorted(ACCOUNTS))
def test_an_instance_that_never_reports_an_endpoint_is_still_destroyed(tmp_path, monkeypatch, provider):
    account = ACCOUNTS[provider](boot=10 ** 9)
    outcome, error = rent(tmp_path, account, monkeypatch)
    assert outcome == 'Unreachable' and 'did not start within its start_minutes' in str(error)
    assert account.created == machines.RENT_ATTEMPTS and account.peak == 1 and account.owned() == []
    assert json.loads((tmp_path / 'job' / 'cleanup.json').read_text())['status'] == 'confirmed'


def test_an_instance_the_provider_made_interruptible_is_refused_and_destroyed(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    answer = account.answer

    def interruptible(command):
        result = answer(command)
        for item in account.instances.values():
            item['is_bid'] = item['label'] != 'someone-else'
        return result
    account.answer = interruptible
    outcome, error = rent(tmp_path, account, monkeypatch)
    assert outcome == 'JobError' and 'interruptible' in str(error)
    assert account.created == 1 and account.owned() == []


@pytest.mark.parametrize('switch', [{'expired': True}, {'can_end_rental': False}, {'watcher_age': 400}, {'deadline': 50}])
def test_a_machine_whose_dead_man_switch_is_not_armed_is_released_before_any_work(tmp_path, monkeypatch, switch):
    account = ACCOUNTS['vastai']()
    outcome, error = rent(tmp_path, account, monkeypatch, machine=lambda address, settings: Machine(address, settings, switch))
    assert outcome == 'Unarmed' and 'dead-man switch is not armed' in str(error)
    assert account.created == machines.RENT_ATTEMPTS and account.peak == 1 and account.owned() == []


@pytest.mark.parametrize('provider', sorted(ACCOUNTS))
def test_every_rented_machine_is_created_with_the_dead_man_switch_as_its_startup(tmp_path, monkeypatch, provider):
    account = ACCOUNTS[provider]()
    kept = {}
    answer = account.answer

    def keeping(command):
        result = answer(command)
        kept.update({key: dict(value) for key, value in vars(account).get('instances', vars(account).get('servers')).items()})
        return result
    account.answer = keeping
    rent(tmp_path, account, monkeypatch)
    startup = next(item['startup'] for item in kept.values() if 'startup' in item)
    assert startup.startswith('#!/bin/bash\n') and machines.DEADMAN_SCRIPT.read_text() in startup
    assert startup.endswith('%s arm %s %d %d\n' % (machines.DEADMAN, provider, machines.STARTUP_SECONDS, 24 * 3600))
    assert len(startup.encode()) <= 4000


def test_a_released_machine_is_released_again_without_error(tmp_path, monkeypatch):
    account = ACCOUNTS['scaleway']()
    rent(tmp_path, account, monkeypatch)
    config = configuration(tmp_path, 'scaleway')
    assert machines.release(config, tmp_path / 'job', sleep=lambda _: None) == 'confirmed'
    assert machines.release(config, tmp_path / 'job', sleep=lambda _: None) == 'confirmed'


def test_children_a_deleted_server_left_behind_are_deleted_from_the_ledger(tmp_path, monkeypatch):
    account = ACCOUNTS['scaleway']()
    account.orphans = True
    outcome, _ = rent(tmp_path, account, monkeypatch)
    assert outcome == 'connected' and account.owned() == []
    assert_nothing_stranded(tmp_path, account, 'orphaned children')


def killed(config, job, account, monkeypatch, machine=Machine):
    """A controller that dies while it holds its machine: no release of its own runs."""
    with monkeypatch.context() as dead, pytest.raises(Died):
        dead.setattr(machines, 'release', lambda *args: None)
        with machines.connect(config, job, machine, sleep=lambda _: None):
            raise Died()


def prepared(tmp_path, account, monkeypatch):
    monkeypatch.setattr(machines, 'provider', lambda provision: account.client())
    config = configuration(tmp_path, account.name)
    job = tmp_path / 'job'
    job.mkdir()
    Resources(config['provisioning']['registry'], account.client()).reap()
    return config, job


def attempts(job):
    return json.loads((job / 'rental.json').read_text())['attempts']


def test_a_resumed_controller_recovers_the_same_machine_and_never_rents_another(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    config, job = prepared(tmp_path, account, monkeypatch)
    killed(config, job, account, monkeypatch)
    assert len(account.owned()) == 1 and attempts(job)[0]['outcome'] is None
    with machines.connect(config, job, Machine, sleep=lambda _: None) as (remote, supervisor):
        assert remote.address == '203.0.113.7'
    assert account.created == 1 and account.owned() == []
    assert [item['outcome'] for item in attempts(job)] == ['used']


@pytest.mark.parametrize('provider', sorted(ACCOUNTS))
def test_a_controller_lost_between_the_lease_and_the_creation_takes_the_lease_up_again(tmp_path, monkeypatch, provider):
    account = ACCOUNTS[provider]()
    config, job = prepared(tmp_path, account, monkeypatch)
    provision = config['provisioning']
    lease = Resources(provision['registry'], account.client()).intent(provision['specification'], machines.LEASE_SECONDS, 3600)
    resources.write(job / 'resource.json', lease)
    machines.Rental(job).begin(lease['token'])
    with machines.connect(config, job, Machine, sleep=lambda _: None):
        assert len(account.owned()) >= 1
    assert account.created == 1 and account.owned() == [] and len(attempts(job)) == 1


@pytest.mark.parametrize('replaceable', [True, False])
def test_a_held_machine_the_reaper_took_is_replaced_only_before_the_run_is_on_it(tmp_path, monkeypatch, replaceable):
    account = ACCOUNTS['vastai']()
    config, job = prepared(tmp_path, account, monkeypatch)
    killed(config, job, account, monkeypatch)
    registry, late = config['provisioning']['registry'], time.time() + machines.LEASE_SECONDS + 1
    Resources(registry, account.client(), clock=lambda: late).reap()
    assert account.owned() == []
    Resources(registry, account.client()).reap()
    if replaceable:
        with machines.connect(config, job, Machine, sleep=lambda _: None):
            assert [item['machine_id'] for item in account.owned()] == [70]
        assert [item['outcome'] for item in attempts(job)] == ['lost', 'used'] and account.created == 2
    else:
        with pytest.raises(resources.ResourceError), machines.connect(config, job, Machine, sleep=lambda _: None, replaceable=False):
            pytest.fail('a run moved to another machine')
        assert account.created == 1
    assert account.peak == 1 and account.owned() == []


class Silent(Machine):
    def command(self, *args, **kwargs):
        raise machines.JobError('connection refused')


def first_silent(count):
    """Machines of which the first `count` never answer."""
    made = []

    def machine(address, settings):
        made.append(address)
        return (Silent if len(made) <= count else Machine)(address, settings)
    return machine


def test_a_machine_that_never_answers_is_released_passed_over_and_replaced_by_the_next_offer(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    outcome, _ = rent(tmp_path, account, monkeypatch, first_silent(1))
    assert outcome == 'connected' and account.created == 2 and account.peak == 1 and account.owned() == []
    assert set(json.loads((tmp_path / 'registry' / 'unreachable.hosts').read_text())) == {'70'}
    first, second = attempts(tmp_path / 'job')
    assert (first['outcome'], first['host'], first['price_per_hour'], first['cleanup']) == ('unusable', '70', 3.0, 'confirmed')
    assert 'did not become reachable' in first['reason'] and first['ended_at'] is not None
    assert (second['outcome'], second['host'], second['price_per_hour'], second['cleanup']) == ('used', '71', 3.2, 'confirmed')
    assert first['token'] != second['token'] and first['instance'] != second['instance']
    assert_nothing_stranded(tmp_path, account, 'replaced', most=2)


def test_a_job_rents_a_bounded_number_of_machines_and_avoids_each(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    outcome, error = rent(tmp_path, account, monkeypatch, Silent)
    assert outcome == 'Unreachable' and account.created == machines.RENT_ATTEMPTS and account.peak == 1 and account.owned() == []
    assert set(json.loads((tmp_path / 'registry' / 'unreachable.hosts').read_text())) == {'70', '71', '72'}
    assert [item['outcome'] for item in attempts(tmp_path / 'job')] == ['unusable'] * machines.RENT_ATTEMPTS


def test_the_bound_on_machines_counts_those_an_earlier_controller_rented(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    config, job = prepared(tmp_path, account, monkeypatch)
    with monkeypatch.context() as earlier, pytest.raises(machines.Unreachable):
        earlier.setattr(machines, 'RENT_ATTEMPTS', 1)
        with machines.connect(config, job, Silent, sleep=lambda _: None):
            pytest.fail('connected')
    assert account.created == 1
    with pytest.raises(machines.Unreachable), machines.connect(config, job, Silent, sleep=lambda _: None):
        pytest.fail('connected')
    assert account.created == machines.RENT_ATTEMPTS and account.peak == 1 and account.owned() == []


def test_no_other_machine_is_rented_while_the_last_is_not_confirmed_gone(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    answer = account.answer

    def undestroyable(command):
        return None if command[2:4] == ['destroy', 'instance'] else answer(command)
    account.answer = undestroyable
    outcome, error = rent(tmp_path, account, monkeypatch, Silent)
    assert outcome == 'JobError' and 'not confirmed released, so no other machine is rented' in str(error)
    assert account.created == 1 and attempts(tmp_path / 'job')[0]['cleanup'] == 'unconfirmed'
    account.answer = answer
    assert_nothing_stranded(tmp_path, account, 'unconfirmed release')


@pytest.mark.parametrize('provider', sorted(ACCOUNTS))
@pytest.mark.parametrize('fault', FAULTS)
def test_no_provider_failure_point_of_a_replaced_rental_strands_or_doubles_an_instance(tmp_path, monkeypatch, provider, fault):
    clean = ACCOUNTS[provider]()
    assert rent(tmp_path / 'clean', clean, monkeypatch, first_silent(1))[0] == 'connected'
    for index in range(len(clean.calls)):
        root = tmp_path / str(index)
        account = ACCOUNTS[provider]()
        account.fail(index, fault)
        rent(root, account, monkeypatch, first_silent(1))
        assert_nothing_stranded(root, account, '%s at provider call %d %r' % (fault, index, clean.calls[index][2:8]), most=2)


def test_an_offer_search_that_finds_nothing_waits_and_gives_up_with_every_lease_released(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    account.offers = []
    config, job = prepared(tmp_path, account, monkeypatch)
    slept = []
    with pytest.raises(resources.NoOffer, match='no matching offer in 13 searches over 60 minutes'), \
            machines.connect(config, job, Machine, sleep=slept.append):
        pytest.fail('connected')
    assert slept == [machines.OFFER_WAIT_SECONDS] * machines.OFFER_WAITS
    assert account.created == 0 and not any(command[2:4] == ['create', 'instance'] for command in account.calls)
    # Nothing was created, so each lease is released at once instead of doubted.
    leases = [json.loads(path.read_text()) for path in (tmp_path / 'registry').glob('*.json') if path.name != 'reaper.json']
    assert len(leases) == machines.OFFER_WAITS + 1 and all(lease['state'] == 'released' for lease in leases)
    assert json.loads((job / 'cleanup.json').read_text())['status'] == 'confirmed'
    assert all((item['outcome'], item['instance'], item['cleanup']) == ('no_offer', None, 'confirmed') for item in attempts(job))
    assert Resources(tmp_path / 'registry', account.client()).unsettled() == {'leases': [], 'instances': []}


def test_an_offer_that_returns_while_the_job_waits_is_rented(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    offers, account.offers = account.offers, []
    config, job = prepared(tmp_path, account, monkeypatch)
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        if slept.count(machines.OFFER_WAIT_SECONDS) == 2:
            account.offers = offers
    with machines.connect(config, job, Machine, sleep=sleep):
        assert len(account.owned()) == 1
    assert [item['outcome'] for item in attempts(job)] == ['no_offer', 'no_offer', 'used'] and account.created == 1


def test_a_job_refuses_to_rent_over_a_lease_whose_release_is_not_confirmed(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    config, job = prepared(tmp_path, account, monkeypatch)
    ledger = Resources(tmp_path / 'registry', account.client())
    doubted = ledger.intent(config['provisioning']['specification'], machines.LEASE_SECONDS, 3600)
    account.fail(len(account.calls) + 2, 'lost')
    with pytest.raises(resources.ResourceError):
        ledger.allocate(doubted['token'], 'startup')
    account.instances.clear()
    with pytest.raises(resources.ResourceError):
        ledger.release(doubted['token'])
    account.heal()
    calls = len(account.calls)
    with pytest.raises(machines.Unsettled, match=doubted['token'] + r' \(cleanup_unconfirmed\)'), \
            machines.connect(config, job, Machine, sleep=lambda _: None):
        pytest.fail('connected')
    assert not (job / 'resource.json').exists() and not any(command[2:4] == ['create', 'instance'] for command in account.calls[calls:])


def test_a_job_settles_a_lapsed_lease_before_it_rents(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    config, job = prepared(tmp_path, account, monkeypatch)
    abandoned = tmp_path / 'abandoned'
    abandoned.mkdir()
    killed(config, abandoned, account, monkeypatch)
    token = json.loads((abandoned / 'resource.json').read_text())['token']
    path = tmp_path / 'registry' / (token + '.json')
    lease = json.loads(path.read_text())
    resources.write(path, dict(lease, expires_at=0))
    with machines.connect(config, job, Machine, sleep=lambda _: None):
        assert [item['label'] for item in account.owned()] == ['pw-training-' + json.loads((job / 'resource.json').read_text())['token']]
    assert account.created == 2 and account.peak == 1 and json.loads(path.read_text())['state'] == 'released'


def test_a_job_refuses_to_rent_beside_a_labelled_instance_no_live_lease_owns(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    config, job = prepared(tmp_path, account, monkeypatch)
    account.instances[5] = {'id': 5, 'label': 'pw-training-' + 'f' * 32, 'listed': 99, 'is_bid': False, 'machine_id': 9, 'dph_total': 1.0}
    with pytest.raises(machines.Unsettled, match=r'labelled instances no live lease owns: 5 \(running\)'), \
            machines.connect(config, job, Machine, sleep=lambda _: None):
        pytest.fail('connected')
    assert account.created == 0 and 5 in account.instances


def test_another_jobs_live_machine_does_not_keep_a_job_from_renting(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    config, job = prepared(tmp_path, account, monkeypatch)
    other = tmp_path / 'other'
    other.mkdir()
    with machines.connect(config, other, Machine, sleep=lambda _: None):
        account.peak = 0
        with machines.connect(config, job, Machine, sleep=lambda _: None):
            assert len(account.owned()) == 2
    assert account.owned() == []
