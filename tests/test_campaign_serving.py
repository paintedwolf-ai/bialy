import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bialy.campaign.journal import CampaignFault
from bialy.campaign.serving import Serving
from bialy.provenance import file_sha256

CONTEXT = {'max_sequence_tokens': 65536, 'thinking': False, 'truncation': 'forbidden'}


TEMPLATE = 't' * 64


def fixture(tmp_path, served=TEMPLATE):
    """A serving stage whose host rendering check reports `served`, or fails when it is None."""
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'model')
    cfg = {'image': 'sha256:server', 'name': 'private', 'gateway': '127.0.0.1', 'checkout': str(tmp_path / 'host'),
           'llama_revision': 'c' * 40}
    calls, container = [], {'state': None}
    def run(args, **kwargs):
        calls.append(args)
        if args[1].endswith('serving_check.py'):
            if served is None:
                raise subprocess.CalledProcessError(1, args, stderr='server renders a conversation differently from training')
            Path(args[args.index('--out') + 1]).write_text(json.dumps({'template_sha256': served}))
        elif args[1] == 'create':
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
    with serving.session(model, file_sha256(model), CONTEXT, 'student', TEMPLATE) as url:
        assert url.startswith('http://127.0.0.1:') and url.endswith('/v1')
        argv = calls[0]
        assert argv[argv.index('--ctx-size') + 1] == '65536'
        assert argv[argv.index('--reasoning') + 1] == 'off'
        assert '--no-context-shift' in argv and '--jinja' in argv and '--chat-template-file' not in argv
        check = next(call for call in calls if call[1].endswith('serving_check.py'))
        assert check[1] == str(tmp_path / 'host/scripts/bialy/custom_model/serving_check.py')
        assert check[check.index('--url') + 1] + '/v1' == url and check[check.index('--revision') + 1] == 'c' * 40
        assert json.loads((serving.directory / 'ready.json').read_text())['container_id'] == 'exact-container'
    assert container['state'] is None
    assert ['docker', 'stop', '--time', '30', 'exact-container'] in calls
    assert json.loads((serving.directory / 'stopped.json').read_text())['container_id'] == 'exact-container'


def test_unrelated_container_is_never_started_or_stopped(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    container['state'] = {'Image': 'sha256:other', 'Config': {'Labels': {}}, 'State': {'Status': 'running'}}
    with pytest.raises(CampaignFault, match='serving_container_binding_mismatch'):
        with serving.session(model, file_sha256(model), CONTEXT, 'student', TEMPLATE):
            pytest.fail('accepted unrelated server')
    assert calls == []


def test_failed_evaluation_cleans_up_its_server(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    with pytest.raises(ValueError, match='evaluation failed'):
        with serving.session(model, file_sha256(model), CONTEXT, 'student', TEMPLATE):
            raise ValueError('evaluation failed')
    assert container['state'] is None


def test_restart_reuses_matching_running_server_and_rejects_changed_artifact(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    plan = serving.plan(model, file_sha256(model), CONTEXT, 'student', TEMPLATE)
    serving.start(plan)
    serving.start(serving.plan(model, file_sha256(model), CONTEXT, 'student', TEMPLATE))
    assert sum(call[1] == 'create' for call in calls) == 1
    assert sum(call[1] == 'start' for call in calls) == 1
    model.write_bytes(b'changed')
    with pytest.raises(CampaignFault, match='serving_inputs_changed'):
        serving.plan(model, file_sha256(model), CONTEXT, 'student', TEMPLATE)
    serving.stop(plan)


def test_unhealthy_server_returns_typed_failure_and_still_stops(tmp_path):
    serving, model, calls, container = fixture(tmp_path)
    serving.configuration['startup_timeout_seconds'] = 0
    with pytest.raises(CampaignFault, match='student_server_startup_failed'):
        with serving.session(model, file_sha256(model), CONTEXT, 'student', TEMPLATE):
            pytest.fail('unhealthy server used')
    assert container['state'] is None


def test_firewall_rule_is_narrow_and_only_removes_its_exact_rule(tmp_path):
    from types import SimpleNamespace
    calls = []
    present = False
    def process(args, **kwargs):
        nonlocal present
        calls.append(args)
        if '-C' in args:
            return SimpleNamespace(returncode=0 if present else 1)
        present = '-I' in args
        return SimpleNamespace(returncode=0)
    server = Serving(tmp_path, {'name': 'bialy', 'gateway': '172.30.0.1'}, run=process)
    plan = {'port': 12345, 'name': 'bialy-student-one'}
    server.access(plan, True)
    server.access(plan, True)
    server.access(plan, False)
    assert sum('-I' in call for call in calls) == 1
    assert sum('-D' in call for call in calls) == 1
    assert all('br-bialy' in call and '12345' in call and 'bialy-student-one' in call for call in calls)


@pytest.mark.parametrize('served,fault', [(None, 'serving_rendering_mismatch'), ('s' * 64, 'serving_template_mismatch')])
def test_a_server_that_does_not_render_as_training_did_takes_no_request(tmp_path, served, fault):
    serving, model, calls, container = fixture(tmp_path, served)
    with pytest.raises(CampaignFault, match=fault):
        with serving.session(model, file_sha256(model), CONTEXT, 'student', TEMPLATE):
            pytest.fail('evaluated against a server that renders differently')
    assert container['state'] is None and not (serving.directory / 'ready.json').exists()
    assert ['docker', 'stop', '--time', '30', 'exact-container'] in calls


def test_both_arms_are_served_by_the_same_command_but_for_the_weights(tmp_path):
    plans = {}
    for arm in ('base', 'trained'):
        (tmp_path / arm).mkdir()
        serving, model, calls, _ = fixture(tmp_path / arm)
        plans[arm] = serving.plan(model, file_sha256(model), CONTEXT, 'student', TEMPLATE)['arguments']
    def comparable(arguments):
        fixed = list(arguments)
        for flag in ('--name', '--label', '-v', '--port'):
            fixed[fixed.index(flag) + 1] = flag
        return fixed
    assert comparable(plans['base']) == comparable(plans['trained'])
    with pytest.raises(CampaignFault, match='serving_inputs_changed'):
        serving.plan(model, file_sha256(model), CONTEXT, 'student', 's' * 64)


def test_an_arm_whose_artifact_is_in_another_template_is_never_served(tmp_path):
    from bialy.campaign.handlers import ServedEvaluation
    specification = {'models': {'e4b': {'directory': 'gemma'}}, 'rendering': {'template_sha256': TEMPLATE}, 'context': CONTEXT}
    stage = ServedEvaluation(SimpleNamespace(directory=tmp_path), None, None, specification, {}, 'e4b', 'base')
    artifact = {'directory': str(tmp_path), 'files': {'model-Q4_K_M.gguf': 'f' * 64}, 'template_sha256': 's' * 64}
    with pytest.raises(CampaignFault, match='evaluation_artifact_template_mismatch'):
        stage.execute({'artifact': artifact}, 'attempt')
