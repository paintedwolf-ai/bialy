import json
from types import SimpleNamespace

import pytest

from bialy.student_host import machines, reaper
from bialy.student_host.jobs import JobError
from bialy.student_host.remote_training import Remote, bootstrap


def configuration(tmp_path):
    return {'host_repository': 'https://example.invalid/host.git', 'host_commit': 'a' * 40,
            'bialy_repository': 'https://example.invalid/bialy.git', 'bialy_commit': 'b' * 40,
            'llama_repository': 'https://example.invalid/llama.cpp.git', 'llama_commit': 'c' * 40,
            'host_checkout': str(tmp_path / 'host'), 'models': str(tmp_path / 'models'),
            'machine': {'host': 'training.example.org', 'user': 'trainer', 'port': 2222,
                        'workspace': '/srv/training', 'uv': '/opt/uv/bin/uv', 'device': 'cuda:1'}}


def test_existing_machine_never_enters_instance_provisioning_or_cleanup(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    machines.validate(config)
    monkeypatch.setattr(machines, 'Resources', lambda *args: pytest.fail('existing machine used provisioning'))
    connections = []
    def remote(host, settings):
        connections.append((host, settings))
        return 'connection'
    with machines.connect(config, tmp_path, False, remote, None) as (connection, check):
        assert connection == 'connection' and check() is None
    assert connections == [('training.example.org', config['machine'])]
    assert not (tmp_path / 'resource.json').exists()
    path = tmp_path / 'machine.json'
    path.write_text(json.dumps(config))
    assert reaper.install(path, run=lambda *args, **kwargs: pytest.fail('installed a reaper')) == {'status': 'not_required'}
    assert reaper.reap(path, once=True) is None


def test_ssh_transport_uses_the_configured_account_port_and_key(tmp_path):
    config = configuration(tmp_path)
    key = tmp_path / 'key'
    key.write_text('fixture')
    config['machine']['identity_file'] = str(key)
    machines.validate(config)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='ready')
    remote = Remote(config['machine']['host'], config['machine'], run=run)
    assert remote.command(['printf', '%s', 'value with spaces']) == 'ready'
    assert ['-p', '2222', '-i', str(key)] == calls[0][-6:-2]
    assert calls[0][-2] == 'trainer@training.example.org'
    assert calls[0][-1] == "printf %s 'value with spaces'"


def test_bootstrap_scopes_sources_tools_and_models_to_the_job_workspace(tmp_path):
    config = configuration(tmp_path)
    calls = []
    bootstrap(SimpleNamespace(command=lambda args, **kwargs: calls.append(args)), config, '/srv/training/job', '26b-a4b')
    clones = [args for args in calls if args[:2] == ['git', 'clone']]
    assert [args[-1] for args in clones] == ['/srv/training/job/host', '/srv/training/job/bialy', '/srv/training/job/llama.cpp']
    assert all('/root/' not in ' '.join(args) and '/scratch/' not in ' '.join(args) for args in calls)
    assert any('UV_TOOL_DIR=/srv/training/job/tools' in args for args in calls)
    models = next(args for args in calls if 'models' in args)
    assert models[-3:] == ['/srv/training/job/models', '--student', '26b-a4b']
    llama = '/srv/training/job/llama.cpp'
    assert ['git', '-C', llama, 'checkout', '--detach', 'c' * 40] in calls
    configure, build = [args for args in calls if args[0] == 'cmake']
    assert '-DGGML_CUDA=OFF' in configure
    assert build[:3] == ['cmake', '--build', llama + '/build'] and {'llama-quantize', 'llama-completion'} <= set(build)


@pytest.mark.parametrize('field,value', [('workspace', '/'), ('workspace', '/srv/../other'), ('user', '-root'), ('port', 0), ('host', '-option')])
def test_invalid_machine_connection_is_rejected_before_execution(tmp_path, field, value):
    config = configuration(tmp_path)
    config['machine'][field] = value
    with pytest.raises(JobError):
        machines.validate(config)


def test_receive_never_pulls_excluded_artifacts(tmp_path):
    calls = []
    remote = Remote('192.0.2.1', configuration(tmp_path)['machine'], run=lambda args, **kwargs: calls.append((args, kwargs)))
    remote.receive('/w/job/', tmp_path / 'job', exclude=('/merged/', '/gguf/'), timeout=7200)
    args, kwargs = calls[0]
    assert args[args.index('/merged/') - 1] == '--exclude' and args[args.index('/gguf/') - 1] == '--exclude'
    assert '.pending-*' in args and kwargs['timeout'] == 7200


def test_export_tools_must_be_pinned_to_a_full_commit(tmp_path):
    config = configuration(tmp_path)
    config['llama_commit'] = 'master'
    with pytest.raises(JobError, match='full commit'):
        machines.validate(config)
