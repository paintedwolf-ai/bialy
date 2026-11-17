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
        with machines.connect(config, job, False, machine, sleep=lambda _: None) as (remote, supervisor):
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


def assert_nothing_stranded(root, account, where):
    cleanup = root / 'job' / 'cleanup.json'
    if cleanup.exists() and json.loads(cleanup.read_text())['status'] == 'confirmed':
        assert account.owned() == [], 'release was confirmed while the account still held resources: %s' % where
    leases = reap_until_settled(root, account)
    assert account.created <= 1, 'a second instance was created: %s' % where
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
    assert outcome == 'JobError' and 'did not become ready' in str(error)
    assert account.created == 1 and account.owned() == []
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
    assert outcome == 'JobError' and 'dead-man switch is not armed' in str(error)
    assert account.created == 1 and account.owned() == []


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


def test_a_resumed_controller_recovers_the_same_machine_and_never_rents_another(tmp_path, monkeypatch):
    account = ACCOUNTS['vastai']()
    monkeypatch.setattr(machines, 'provider', lambda provision: account.client())
    config = configuration(tmp_path, 'vastai')
    job = tmp_path / 'job'
    job.mkdir()
    Resources(config['provisioning']['registry'], account.client()).reap()
    with monkeypatch.context() as killed, pytest.raises(Died):
        # A killed controller runs no release of its own.
        killed.setattr(machines, 'release', lambda *args: None)
        with machines.connect(config, job, False, Machine, sleep=lambda _: None):
            raise Died()
    assert len(account.owned()) == 1
    with machines.connect(config, job, True, Machine, sleep=lambda _: None) as (remote, supervisor):
        assert remote.address == '203.0.113.7'
    assert account.created == 1 and account.owned() == []


def test_a_machine_that_never_answers_is_released_and_the_next_rental_lands_elsewhere(tmp_path, monkeypatch):
    class Silent(Machine):
        def command(self, *args, **kwargs):
            raise machines.JobError('connection refused')

    account = ACCOUNTS['vastai']()
    outcome, _ = rent(tmp_path / 'first', account, monkeypatch, Silent)
    assert outcome == 'Unreachable' and account.owned() == []
    registry = tmp_path / 'first' / 'registry'
    assert set(json.loads((registry / 'unreachable.hosts').read_text())) == {'70'}
    monkeypatch.setattr(machines, 'provider', lambda provision: account.client())
    config = configuration(tmp_path / 'first', 'vastai')
    job = tmp_path / 'first' / 'again'
    job.mkdir()
    with machines.connect(config, job, False, Machine, sleep=lambda _: None):
        assert [item['machine_id'] for item in account.owned()] == [71]
    assert account.owned() == []
