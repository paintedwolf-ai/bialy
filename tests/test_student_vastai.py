import contextlib
import json
import time
from types import SimpleNamespace

import pytest

from bialy.student_host import machines, reaper
from bialy.student_host.resources import ResourceError, write
from bialy.student_host.vastai import VastAI


def test_vastai_rejects_unowned_and_unstructured_responses():
    responses = iter([
        {'unexpected': []},
        [{'id': 101, 'label': 'foreign-instance', 'tags': ['token'], 'ssh_host': 'ssh5.vast.ai', 'ssh_port': 41234}],
        [{'id': 102, 'label': 'pw-training-token', 'ssh_host': '', 'ssh_port': 41234}],
    ])

    def run(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(next(responses)))

    provider = VastAI(run=run)
    with pytest.raises(ResourceError, match='shape'):
        provider.find('token')
    with pytest.raises(ResourceError, match='ownership'):
        provider.find('token')
    with pytest.raises(ResourceError, match='SSH connection'):
        provider.find('token')


def test_vastai_create_delete_and_dynamic_ssh_port_connect(tmp_path, monkeypatch):
    registry = tmp_path / 'registry'
    registry.mkdir()
    write(registry / 'reaper.json', {'at': time.time(), 'pid': 1})

    instances = []
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert args[:2] == ['vastai', '--raw']
        cmd = args[2:]
        if cmd[:2] == ['show', 'instances']:
            return SimpleNamespace(returncode=0, stdout=json.dumps(instances))
        if cmd[:2] == ['search', 'offers']:
            assert 'gpu_name=H100_SXM' in cmd[-1]
            assert 'dph_total<=1.10' in cmd[-1]
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'id': 777}]))
        if cmd[:3] == ['create', 'instance', '777']:
            label = cmd[cmd.index('--label') + 1]
            assert '--bid_price' in cmd and cmd[cmd.index('--bid_price') + 1] == '1.10'
            instances.append({
                'id': 9001,
                'label': label,
                'ssh_host': 'ssh5.vast.ai',
                'ssh_port': 41234,
            })
            return SimpleNamespace(returncode=0, stdout=json.dumps({'success': True, 'new_contract': 9001}))
        if cmd[:3] == ['destroy', 'instance', '9001']:
            instances.clear()
            return SimpleNamespace(returncode=0, stdout=json.dumps({'success': True}))
        raise AssertionError('unexpected command: %r' % (args,))

    monkeypatch.setattr(reaper, 'VastAI', lambda api_key_file=None: VastAI(api_key_file, run=run))
    config = {
        'host_repository': 'https://example.invalid/host.git',
        'host_commit': 'a' * 40,
        'bialy_repository': 'https://example.invalid/bialy.git',
        'bialy_commit': 'b' * 40,
        'llama_repository': 'https://example.invalid/llama.cpp.git',
        'llama_commit': 'c' * 40,
        'host_checkout': str(tmp_path / 'host'),
        'models': str(tmp_path / 'models'),
        'provisioning': {
            'provider': 'vastai',
            'registry': str(registry),
            'specification': {
                'gpu_name': 'H100_SXM',
                'num_gpus': '1',
                'image': 'pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel',
                'disk': '250',
                'bid_price': '1.10',
            },
        },
        'machine': {
            'user': 'root',
            'workspace': '/scratch/training',
            'uv': '/root/.local/bin/uv',
        },
    }
    machines.validate(config)
    connected = []

    @contextlib.contextmanager
    def dummy_renewal(resources, token, seconds):
        yield SimpleNamespace(check=lambda: None)

    def dummy_remote(host, settings):
        connected.append((host, settings))
        return 'vast-remote'

    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    with machines.connect(config, job_dir, False, dummy_remote, dummy_renewal) as (conn, _):
        assert conn == 'vast-remote'
    assert connected == [('ssh5.vast.ai', {'user': 'root', 'workspace': '/scratch/training', 'uv': '/root/.local/bin/uv', 'port': 41234})]
    assert json.loads((job_dir / 'cleanup.json').read_text())['status'] == 'confirmed'
    assert instances == []
