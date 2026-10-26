import json

import pytest

from bialy.student_host.resources import ResourceError, Resources


class Provider:
    def __init__(self):
        self.servers = []
        self.available = True
        self.crash = False

    def find(self, token):
        if not self.available:
            raise ConnectionError('offline')
        return [server for server in self.servers if token in server['tags']]

    def create(self, token, specification):
        server = {'id': 'owned', 'tags': [token]}
        self.servers.append(server)
        if self.crash:
            raise ConnectionError('response lost')
        return server

    def delete(self, server):
        if server in self.servers:
            self.servers.remove(server)

    def absent(self, server):
        return server not in self.servers


def test_create_response_loss_reconciles_without_duplicate_resource(tmp_path):
    provider = Provider()
    resources = Resources(tmp_path, provider, clock=lambda: 100)
    resources.reap()
    intent = resources.intent({'type': 'gpu'}, 120)
    provider.crash = True
    with pytest.raises(ConnectionError):
        resources.allocate(intent)
    provider.crash = False
    assert resources.allocate(intent)['server']['id'] == 'owned'
    assert len(provider.servers) == 1
    resources.release(intent['token'])
    assert not provider.servers


def test_reaper_cleans_crashed_allocation_and_preserves_foreign_resources(tmp_path):
    provider = Provider()
    resources = Resources(tmp_path, provider, clock=lambda: 100)
    resources.reap()
    intent = resources.intent({}, 120)
    provider.crash = True
    with pytest.raises(ConnectionError):
        resources.allocate(intent)
    provider.servers.append({'id': 'foreign', 'tags': ['someone-else']})
    results = Resources(tmp_path, provider, clock=lambda: 221).reap()
    assert results[0]['state'] == 'released'
    assert provider.servers == [{'id': 'foreign', 'tags': ['someone-else']}]


def test_control_plane_failure_does_not_claim_deletion(tmp_path):
    provider = Provider()
    resources = Resources(tmp_path, provider, clock=lambda: 100)
    with pytest.raises(ResourceError, match='reaper'):
        resources.intent({}, 120)
    resources.reap()
    intent = resources.intent({}, 120)
    resources.allocate(intent)
    provider.available = False
    with pytest.raises(ConnectionError):
        resources.release(intent['token'])
    assert json.loads((tmp_path / (intent['token'] + '.json')).read_text())['state'] == 'cleanup_unconfirmed'
    provider.available = True
    assert resources.reap()[0]['state'] == 'released'


def test_delayed_allocation_remains_owned_until_observed_and_removed(tmp_path):
    provider = Provider()
    resources = Resources(tmp_path, provider, clock=lambda: 100)
    resources.reap()
    intent = resources.intent({}, 120)
    provider.crash = True
    with pytest.raises(ConnectionError):
        resources.allocate(intent)
    delayed = provider.servers.pop()
    with pytest.raises(ResourceError, match='unknown'):
        resources.allocate(intent)
    with pytest.raises(ResourceError, match='unknown'):
        resources.release(intent['token'])
    provider.servers.append(delayed)
    assert resources.reap()[0]['state'] == 'released'
    assert not provider.servers
    provider.servers.append(delayed)
    assert Resources(tmp_path, provider, clock=lambda: 221).reap()[0]['state'] == 'released'
    assert not provider.servers
