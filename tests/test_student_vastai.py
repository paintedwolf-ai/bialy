import json
import subprocess
from types import SimpleNamespace

import pytest
from rental_accounts import VastAccount

from bialy.student_host.resources import NoOffer, ResourceError
from bialy.student_host.vastai import PINS, VastAI, cli_version, label


def rent(provider, token, startup, avoid=()):
    """Take the cheapest offer, as the ledger does."""
    return provider.create(token, SPECIFICATION, startup, provider.offer(SPECIFICATION, avoid))


SPECIFICATION = {'gpu_name': 'H100_SXM', 'num_gpus': 2, 'image': 'example/image:1', 'max_price': 3.5, 'download_mbps': 500,
                 'gpu_memory_gib': 79, 'memory_gib': 128, 'disk_gib': 320}


DIRECT = {'22/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '41235'}]}


def listing(*instances):
    return VastAI(run=lambda args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(list(instances))))


@pytest.mark.parametrize('state', [
    {}, {'actual_status': 'loading'}, {'actual_status': 'running', 'public_ipaddr': '', 'ports': DIRECT},
    {'actual_status': 'running', 'public_ipaddr': '203.0.113.7', 'ports': {}},
    {'actual_status': 'running', 'ssh_host': 'ssh.example.org', 'ssh_port': 41234},
    {'actual_status': 'exited', 'public_ipaddr': '203.0.113.7', 'ports': DIRECT}, {'actual_status': 'offline'}])
def test_an_owned_instance_is_found_in_any_state_and_has_no_endpoint_until_it_can_be_reached(state):
    found = listing({'id': 7, 'label': label('token'), **state}).owned('token')
    assert [(item['id'], item['endpoint']) for item in found] == [('7', None)]


def test_a_running_instance_is_reached_at_its_own_address_never_through_the_proxy():
    found = listing({'id': 7, 'label': label('token'), 'actual_status': 'running', 'ssh_host': 'ssh.example.org', 'ssh_port': 41234,
                     'public_ipaddr': '203.0.113.7', 'ports': DIRECT, 'machine_id': 70, 'is_bid': True}).owned('token')
    assert found[0]['endpoint'] == {'address': '203.0.113.7', 'port': 41235} and found[0]['interruptible'] is True
    assert found[0]['host'] == '70'


def test_ownership_is_the_exact_label_and_nothing_else():
    provider = listing({'id': 1, 'label': label('token') + 'x'}, {'id': 2, 'label': None}, {'id': 3, 'label': label('other')},
                       {'id': 4, 'label': label('token')})
    assert [item['id'] for item in provider.owned('token')] == ['4']


@pytest.mark.parametrize('stdout', ['{"unexpected": []}', '[{"label": "no identity"}]', 'not json'])
def test_an_unstructured_listing_is_an_error_and_never_an_empty_account(stdout):
    provider = VastAI(run=lambda args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout))
    with pytest.raises(ResourceError):
        provider.owned('token')
    with pytest.raises(ResourceError):
        provider.remaining({'id': '7'})


def test_an_instance_is_rented_on_demand_within_the_declared_capacity_and_price():
    account = VastAccount()
    created = rent(account.client(), 'token', 'startup')
    search, create = account.calls
    assert search[2:6] == ['search', 'offers', '--type', 'on-demand']
    assert search[-1].split() == ['gpu_name=H100_SXM', 'num_gpus=2', 'gpu_ram>=79', 'cpu_ram>=128', 'disk_space>=320',
                                  'inet_down>=500', 'dph_total<=3.5', 'rentable=true', 'verified=true']
    assert not any('bid' in argument for argument in create)
    assert create[create.index('--label') + 1] == label('token') and create[create.index('--disk') + 1] == '320'
    assert create[create.index('--onstart-cmd') + 1] == 'startup'
    assert created == {'id': '9001', 'state': 'pending', 'interruptible': False, 'children': [], 'host': '70', 'price_per_hour': 3.0,
                       'endpoint': None}


def test_creation_needs_no_listing_and_a_startup_the_provider_would_truncate_is_refused():
    account = VastAccount(lag=10 ** 9)
    assert rent(account.client(), 'token', 'startup')['id'] == '9001'
    with pytest.raises(ResourceError, match='limit'):
        rent(account.client(), 'token', 'x' * 4001)
    assert account.created == 1


def test_destroy_is_requested_by_identity_even_when_the_listing_omits_the_instance():
    account = VastAccount(lag=10 ** 9)
    provider = account.client()
    instance = rent(provider, 'token', 'startup')
    assert provider.remaining(instance) == [] and len(account.owned()) == 1
    assert provider.destroy(instance) is True and account.owned() == []
    assert provider.destroy(instance) is False


def test_a_refused_destroy_of_a_listed_instance_is_an_error_and_a_lost_answer_is_not_an_acknowledgement():
    account = VastAccount()
    provider = account.client()
    instance = rent(provider, 'token', 'startup')
    account.fail(len(account.calls), 'refused')
    with pytest.raises(ResourceError):
        provider.destroy(instance)
    assert len(account.owned()) == 1
    account.fail(len(account.calls), 'timeout')
    assert provider.destroy(instance) is False and account.owned() == []


def test_the_account_key_reaches_the_cli_through_its_environment_and_never_its_arguments(tmp_path):
    key = tmp_path / 'key'
    key.write_text('account-secret\n')
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='[]')
    VastAI(str(key), run=run).owned('token')
    args, kwargs = calls[0]
    assert kwargs['env']['VAST_API_KEY'] == 'account-secret'
    assert not any('account-secret' in argument or str(key) in argument for argument in args)


def test_a_lost_answer_is_not_a_success():
    def run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 120)
    with pytest.raises(subprocess.TimeoutExpired):
        VastAI(run=run).owned('token')


def test_destroy_answers_the_confirmation_the_cli_would_otherwise_wait_for():
    account = VastAccount()
    provider = account.client()
    instance = rent(provider, 'token', 'arm')
    provider.destroy(instance)
    assert account.calls[-1][2:] == ['destroy', 'instance', instance['id'], '--yes'] and account.owned() == []


def test_an_offer_on_an_avoided_machine_is_passed_over_for_the_next_cheapest():
    account = VastAccount()
    created = rent(account.client(), 'token', 'startup', {'70'})
    assert account.calls[-1][4] == '778' and created['host'] == '71'
    with pytest.raises(NoOffer, match='no matching on-demand offer'):
        rent(account.client(), 'other', 'startup', {str(offer['machine_id']) for offer in account.offers})


def test_the_offer_states_the_hourly_price_the_instance_is_then_listed_at():
    account = VastAccount()
    provider = account.client()
    offer = provider.offer(SPECIFICATION)
    assert offer == {'id': '777', 'host': '70', 'price_per_hour': 3.0} and len(account.calls) == 1
    assert provider.create('token', SPECIFICATION, 'startup', offer)['price_per_hour'] == 3.0
    account.instances[9001]['dph_total'] = 3.07
    assert provider.owned('token')[0]['price_per_hour'] == 3.07


def test_every_labelled_instance_of_the_account_is_found_by_its_token():
    provider = listing({'id': 1, 'label': label('one')}, {'id': 2, 'label': None}, {'id': 3, 'label': 'someone-else'},
                       {'id': 4, 'label': label('two'), 'actual_status': 'exited'}, {'id': 5, 'label': label('one')})
    found = provider.labelled()
    assert {token: [item['id'] for item in items] for token, items in found.items()} == {'one': ['1', '5'], 'two': ['4']}
    assert found['two'][0]['state'] == 'exited'


def test_the_cli_release_is_pinned_where_setup_installs_it_from():
    setup = (PINS.parent / 'setup.sh').read_text()
    assert 'uv tool install "vastai==$VASTAI_CLI_VERSION"' in setup and cli_version().count('.') == 2


@pytest.mark.parametrize('answer', [SimpleNamespace(returncode=0, stdout='1.0.13\n'), SimpleNamespace(returncode=0, stdout=''),
                                    SimpleNamespace(returncode=2, stdout=cli_version()), FileNotFoundError('vastai'),
                                    subprocess.TimeoutExpired('vastai', 30)])
def test_any_cli_but_the_pinned_release_is_refused_with_the_release_to_install(answer):
    def run(args, **kwargs):
        assert args == ['vastai', '--version']
        if isinstance(answer, Exception):
            raise answer
        return answer
    with pytest.raises(ResourceError, match='vastai==' + cli_version().replace('.', r'\.')):
        VastAI(run=run).check()
    assert VastAI(run=lambda args, **kwargs: SimpleNamespace(returncode=0, stdout=cli_version() + '\n')).check() == cli_version()
