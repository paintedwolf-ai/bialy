import json
from types import SimpleNamespace

import pytest

from bialy.campaign.journal import CampaignFault
from bialy.campaign.serving import Serving
from bialy.provenance import file_sha256

CONTEXT = {'max_sequence_tokens': 65536, 'thinking': False, 'truncation': 'forbidden'}


def fixture(tmp_path):
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'model')
    cfg = {'image': 'sha256:server', 'network': 'private', 'gateway': '127.0.0.1'}
    calls, container = [], {'state': None}
    def run(args, **kwargs):
        calls.append(args)
        if args[1] == 'create':
            label = args[args.index('--label') + 1].split('=', 1)[1]
            container['state'] = {'Id': 'exact-container', 'Image': cfg['image'],
                                  'Config': {'Labels': {'bialy.serving': label}}, 'State': {'Status': 'created'}}
        elif args[1] == 'start':
            container['state']['State']['Status'] = 'running'
        elif args[1] == 'stop':
            container['state']['State']['Status'] = 'exited'
        elif args[1] == 'rm':
            container['state'] = None
        return SimpleNamespace(returncode=0, stdout='')
    serving = Serving(tmp_path / 'serving', cfg, run=run, health=lambda url: True)
    serving.containers.inspect = lambda name: container['state']
    return serving, model, calls, container


def test_student_lifecycle_pins_context_disables_thinking_and_stops_exact_container(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    with serving.session(model, file_sha256(model), CONTEXT, 'student') as url:
        assert url.startswith('http://127.0.0.1:') and url.endswith('/v1')
        argv = calls[0]
        assert argv[argv.index('--ctx-size') + 1] == '65536'
        assert argv[argv.index('--reasoning') + 1] == 'off'
        assert '--no-context-shift' in argv and '--jinja' in argv
    assert container['state'] is None
    assert ['docker', 'stop', '--time', '30', 'exact-container'] in calls
    assert json.loads((serving.directory / 'stopped.json').read_text())['container_id'] == 'exact-container'


def test_unrelated_container_is_never_started_or_stopped(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    container['state'] = {'Image': 'sha256:other', 'Config': {'Labels': {}}, 'State': {'Status': 'running'}}
    with pytest.raises(CampaignFault, match='serving_container_binding_mismatch'):
        with serving.session(model, file_sha256(model), CONTEXT, 'student'):
            pytest.fail('accepted unrelated server')
    assert calls == []


def test_failed_evaluation_cleans_up_its_server(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    with pytest.raises(ValueError, match='evaluation failed'):
        with serving.session(model, file_sha256(model), CONTEXT, 'student'):
            raise ValueError('evaluation failed')
    assert container['state'] is None


def test_restart_reuses_matching_running_server_and_rejects_changed_artifact(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    plan = serving.plan(model, file_sha256(model), CONTEXT, 'student')
    serving.start(plan)
    serving.start(serving.plan(model, file_sha256(model), CONTEXT, 'student'))
    assert sum(call[1] == 'create' for call in calls) == 1
    assert sum(call[1] == 'start' for call in calls) == 1
    model.write_bytes(b'changed')
    with pytest.raises(CampaignFault, match='serving_inputs_changed'):
        serving.plan(model, file_sha256(model), CONTEXT, 'student')
    serving.stop(plan)


def test_unhealthy_server_returns_typed_failure_and_still_stops(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    serving.configuration['startup_timeout_seconds'] = 0
    with pytest.raises(CampaignFault, match='student_server_startup_failed'):
        with serving.session(model, file_sha256(model), CONTEXT, 'student'):
            pytest.fail('unhealthy server used')
    assert container['state'] is None
