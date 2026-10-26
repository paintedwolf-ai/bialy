import json
from types import SimpleNamespace

import pytest

from bialy.student_host.resources import ResourceError
from bialy.student_host.scaleway import Scaleway


def test_provider_rejects_unowned_and_unstructured_responses():
    responses = iter([{'unexpected': []}, [{'id': 'foreign', 'project': 'another', 'tags': ['token']}]])
    def run(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(next(responses)))
    provider = Scaleway('profile', 'project', 'zone', run=run)
    with pytest.raises(ResourceError, match='shape'):
        provider.find('token')
    with pytest.raises(ResourceError, match='ownership'):
        provider.find('token')


def test_deletion_reconciles_owned_children_after_server_is_gone():
    server = {'id': 'server', 'project': 'project', 'tags': ['token'],
              'volumes': {'0': {'id': 'volume'}}, 'public_ips': [{'id': 'address'}]}
    remaining = {('instance', 'server'): [server], ('instance', 'volume'): [],
                 ('block', 'volume'): [{'id': 'volume'}, {'id': 'foreign'}],
                 ('instance', 'ip'): [{'id': 'address'}]}
    deleted = []
    def run(args, **kwargs):
        service, kind, action = args[5:8]
        assert args[-1] == 'zone=zone'
        if action == 'list':
            assert 'project-id=project' in args
            return SimpleNamespace(returncode=0, stdout=json.dumps(remaining[(service, kind)]))
        identifier = args[8]
        deleted.append((service, kind, identifier))
        remaining[(service, kind)] = [item for item in remaining[(service, kind)] if item['id'] != identifier]
        return SimpleNamespace(returncode=0, stdout='')
    provider = Scaleway('profile', 'project', 'zone', run=run)
    provider.delete(server)
    assert provider.absent(server)
    assert deleted == [('instance', 'server', 'server'), ('block', 'volume', 'volume'), ('instance', 'ip', 'address')]
    assert remaining[('block', 'volume')] == [{'id': 'foreign'}]


def test_provider_error_never_means_resource_absent():
    provider = Scaleway('profile', 'project', 'zone', run=lambda *a, **kw: SimpleNamespace(returncode=1, stdout=''))
    with pytest.raises(ResourceError, match='confirm'):
        provider.absent({'id': 'server'})
