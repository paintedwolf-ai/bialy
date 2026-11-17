import json
from types import SimpleNamespace

import pytest
from rental_accounts import ScalewayAccount

from bialy.student_host.resources import ResourceError
from bialy.student_host.scaleway import Scaleway, name, normalize


def rent(provider, token, startup, avoid=()):
    """Take the cheapest offer, as the ledger does."""
    return provider.create(token, SPECIFICATION, startup, provider.offer(SPECIFICATION, avoid))


SPECIFICATION = {'type': 'H100-1-80G', 'image': 'image', 'root_volume': '1TB', 'gpu_memory_gib': 79, 'memory_gib': 200, 'disk_gib': 1000}


def server(**fields):
    return {'id': 'server', 'name': name('token'), 'tags': ['token'], 'project': 'project', 'state': 'running',
            'volumes': {'0': {'id': 'volume'}}, 'public_ips': [{'id': 'address', 'address': '192.0.2.7'}], **fields}


@pytest.mark.parametrize('state', ['starting', 'stopping', 'stopped', 'stopped in place', 'locked'])
def test_a_server_is_found_in_any_state_with_its_children_and_no_endpoint_until_it_runs(state):
    found = normalize(server(state=state))
    assert found['endpoint'] is None and found['children'] == [['ip', 'address'], ['volume', 'volume']]
    assert normalize(server())['endpoint'] == {'address': '192.0.2.7', 'port': 22}
    assert normalize(server(public_ips=[], public_ip=None))['endpoint'] is None


def test_ownership_needs_the_tag_the_name_and_the_project():
    rows = [server(id='ours'), server(id='another-project', project='another'), server(id='renamed', name='other'),
            server(id='untagged', tags=[])]
    provider = Scaleway('profile', 'project', 'zone', run=lambda args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(rows)))
    assert [item['id'] for item in provider.owned('token')] == ['ours']


@pytest.mark.parametrize('stdout', ['{"unexpected": []}', '[{"name": "no identity"}]', 'not json'])
def test_an_unstructured_listing_is_an_error_and_never_an_empty_project(stdout):
    provider = Scaleway('profile', 'project', 'zone', run=lambda args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout))
    with pytest.raises(ResourceError):
        provider.owned('token')


def test_a_server_is_created_with_its_token_and_startup_and_returned_with_its_children():
    account = ScalewayAccount()
    created = rent(account.client(), 'token', '#!/bin/bash\narm\n')
    arguments = account.calls[0]
    assert {'name=' + name('token'), 'tags.0=token', 'type=H100-1-80G', 'root-volume=1TB', 'cloud-init=#!/bin/bash\narm\n'} <= set(arguments)
    assert created['children'] == [['ip', 'server-1-ip'], ['volume', 'server-1-volume']] and created['endpoint'] is None


def test_destroy_deletes_children_the_server_left_behind_and_nothing_foreign():
    account = ScalewayAccount()
    account.orphans = True
    provider = account.client()
    instance = rent(provider, 'token', 'startup')
    assert provider.destroy(instance) is True
    assert provider.remaining(instance) == [] and account.owned() == []
    assert account.children[('block', 'volume')] == {'foreign-volume'} and account.children[('instance', 'ip')] == {'foreign-ip'}
    assert provider.destroy(instance) is False


def test_a_refused_delete_of_a_listed_server_is_an_error_and_a_failed_listing_never_means_absent():
    account = ScalewayAccount()
    provider = account.client()
    instance = rent(provider, 'token', 'startup')
    account.fail(len(account.calls), 'refused')
    with pytest.raises(ResourceError):
        provider.destroy(instance)
    account.fail(len(account.calls), 'outage')
    with pytest.raises(ResourceError):
        provider.remaining(instance)
    assert len(account.owned()) == 3
