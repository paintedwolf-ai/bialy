"""The scripts a rented machine runs, executed in the image the Vast.ai example names.

This starts a local container and never contacts a provider. It runs only where Docker
already holds the image, because the dead-man switch ends by killing every process it can
reach, which belongs inside a container."""
import json
import shutil
import subprocess
import time
import uuid

import pytest

from bialy.student_host import machines

EXAMPLE = json.loads((machines.ROOT / 'train-host/student/vastai-machine.example.json').read_text())
IMAGE = EXAMPLE['provisioning']['specification']['image']
API = '''
import http.server
class Handler(http.server.BaseHTTPRequestHandler):
    def answer(self):
        with open('/tmp/api.log', 'a') as log:
            log.write('%s %s %s\\n' % (self.command, self.path, self.headers['Authorization']))
        self.send_response(500)
        self.end_headers()
    do_DELETE = do_PUT = answer
http.server.HTTPServer(('127.0.0.1', 8099), Handler).serve_forever()
'''


def docker(*arguments, **kwargs):
    return subprocess.run(['docker', *arguments], capture_output=True, text=True, **kwargs)


def held():
    return shutil.which('docker') is not None and docker('image', 'inspect', IMAGE).returncode == 0


@pytest.fixture
def container():
    name = 'pw-machine-image-' + uuid.uuid4().hex[:12]
    started = docker('run', '-d', '--rm', '--name', name, '-e', 'CONTAINER_ID=4242', '-e', 'CONTAINER_API_KEY=instance-key',
                     '-e', 'PW_DEADMAN_API=http://127.0.0.1:8099', IMAGE, 'sleep', 'infinity')
    assert started.returncode == 0, started.stderr
    try:
        yield name
    finally:
        docker('rm', '-f', name)


def switch(container, seconds):
    return json.loads(docker('exec', container, machines.DEADMAN, 'extend', str(seconds)).stdout)


def until(condition, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(2)
    return False


@pytest.mark.skipif(not held(), reason='Docker does not hold the example image')
def test_the_example_image_arms_the_switch_passes_preflight_and_ends_its_own_rental(container):
    provision = {**EXAMPLE['provisioning'], 'lifetime_hours': 1}
    armed = docker('exec', '-i', container, 'bash', '-s', input=machines.startup(provision))
    assert armed.returncode == 0, armed.stderr
    status = switch(container, 600)
    assert status['expired'] is False and status['watcher_age'] <= 30 and status['limit'] - status['now'] <= 3600
    assert status['deadline'] - status['now'] == 600
    assert switch(container, 10 ** 6)['deadline'] == status['limit']
    assert until(lambda: switch(container, 600)['can_end_rental'], 300), 'the image never gained a way to call the provider'

    preflight = docker('exec', '-i', container, 'bash', '-s', '--', 'install', '/workspace/training', '/root/.local/bin/uv', 'cpu', '1', '1', '0',
                       input=machines.PREFLIGHT_SCRIPT.read_text())
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    assert json.loads(preflight.stdout.strip().splitlines()[-1])['problems'] == []
    assert docker('exec', container, '/root/.local/bin/uv', '--version').returncode == 0
    short = docker('exec', '-i', container, 'bash', '-s', '--', 'check', '/workspace/training', '/root/.local/bin/uv', 'cuda:0', '1000000', '1', '1',
                   input=machines.PREFLIGHT_SCRIPT.read_text())
    problems = json.loads(short.stdout.strip().splitlines()[-1])['problems']
    assert short.returncode == 1 and any('GiB free' in problem for problem in problems) and any('GPU' in problem for problem in problems)

    docker('exec', '-d', container, 'python3', '-c', API)
    docker('exec', '-d', container, 'sleep', '31337')
    assert until(lambda: docker('exec', container, 'pgrep', '-f', 'sleep 31337').returncode == 0, 20)
    switch(container, 4)
    assert until(lambda: docker('exec', container, 'pgrep', '-f', 'sleep 31337').returncode != 0, 60), 'the work outlived the deadline'
    log = docker('exec', container, 'cat', '/tmp/api.log').stdout.splitlines()
    assert log[:2] == ['DELETE /instances/4242/ Bearer instance-key', 'PUT /instances/4242/ Bearer instance-key']
    assert switch(container, 600)['expired'] is True
    identity = docker('exec', container, 'stat', '-c', '%a', '/var/lib/pw-deadman/identity').stdout.strip()
    assert identity == '600'
