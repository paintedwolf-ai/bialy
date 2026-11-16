import json
import subprocess
from types import SimpleNamespace

import pytest
from rental_accounts import VastAccount

from bialy.student_host.resources import ResourceError
from bialy.student_host.vastai import VastAI, label

SPECIFICATION = {'gpu_name': 'H100_SXM', 'num_gpus': 2, 'image': 'example/image:1', 'max_price': 3.5,
                 'gpu_memory_gib': 79, 'memory_gib': 128, 'disk_gib': 320}


def listing(*instances):
    return VastAI(run=lambda args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(list(instances))))


@pytest.mark.parametrize('state', [
    {}, {'actual_status': 'loading'}, {'actual_status': 'running', 'ssh_host': '', 'ssh_port': 41234},
    {'actual_status': 'running', 'ssh_host': 'ssh.example.org', 'ssh_port': None},
    {'actual_status': 'exited', 'ssh_host': 'ssh.example.org', 'ssh_port': 41234}, {'actual_status': 'offline'}])
def test_an_owned_instance_is_found_in_any_state_and_has_no_endpoint_until_it_can_be_reached(state):
    found = listing({'id': 7, 'label': label('token'), **state}).owned('token')
    assert [(item['id'], item['endpoint']) for item in found] == [('7', None)]


def test_only_a_running_instance_with_connection_details_has_an_endpoint():
    found = listing({'id': 7, 'label': label('token'), 'actual_status': 'running', 'ssh_host': 'ssh.example.org', 'ssh_port': 41234,
                     'is_bid': True}).owned('token')
    assert found[0]['endpoint'] == {'address': 'ssh.example.org', 'port': 41234} and found[0]['interruptible'] is True


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
    created = account.client().create('token', SPECIFICATION, 'startup')
    search, create = account.calls
    assert search[2:6] == ['search', 'offers', '--type', 'on-demand']
    assert search[-1].split() == ['gpu_name=H100_SXM', 'num_gpus=2', 'gpu_ram>=79', 'cpu_ram>=128', 'disk_space>=320',
                                  'dph_total<=3.5', 'rentable=true', 'verified=true']
    assert not any('bid' in argument for argument in create)
    assert create[create.index('--label') + 1] == label('token') and create[create.index('--disk') + 1] == '320'
    assert create[create.index('--onstart-cmd') + 1] == 'startup'
    assert created == {'id': '9001', 'state': 'pending', 'interruptible': False, 'children': [], 'endpoint': None}


def test_creation_needs_no_listing_and_a_startup_the_provider_would_truncate_is_refused():
    account = VastAccount(lag=10 ** 9)
    assert account.client().create('token', SPECIFICATION, 'startup')['id'] == '9001'
    with pytest.raises(ResourceError, match='limit'):
        account.client().create('token', SPECIFICATION, 'x' * 4001)
    assert account.created == 1


def test_destroy_is_requested_by_identity_even_when_the_listing_omits_the_instance():
    account = VastAccount(lag=10 ** 9)
    provider = account.client()
    instance = provider.create('token', SPECIFICATION, 'startup')
    assert provider.remaining(instance) == [] and len(account.owned()) == 1
    assert provider.destroy(instance) is True and account.owned() == []
    assert provider.destroy(instance) is False


def test_a_refused_destroy_of_a_listed_instance_is_an_error_and_a_lost_answer_is_not_an_acknowledgement():
    account = VastAccount()
    provider = account.client()
    instance = provider.create('token', SPECIFICATION, 'startup')
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
