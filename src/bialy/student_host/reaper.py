"""Independent cleanup service for expired training resource leases."""
import json
import time
from pathlib import Path

from .resources import Resources
from .scaleway import Scaleway


def provider(configuration):
    return Scaleway(configuration['profile'], configuration['project'], configuration['zone'])


def reap(configuration_path, once=False):
    configuration = json.loads(Path(configuration_path).read_text()).get('provisioning')
    if configuration is None:
        return
    resources = Resources(configuration['registry'], provider(configuration))
    while True:
        resources.reap()
        if once:
            return
        time.sleep(20)


def install(configuration_path, run=None):
    import os
    import shutil
    import subprocess
    import tempfile
    run = run or subprocess.run
    configuration_path = Path(configuration_path).resolve(strict=True)
    configuration = json.loads(configuration_path.read_text()).get('provisioning')
    if configuration is None:
        return {'status': 'not_required'}
    if configuration_path.stat().st_mode & 0o077:
        raise ValueError('machine configuration must be private')
    root = Path(__file__).resolve().parents[3]
    uv = shutil.which('uv')
    if uv is None:
        raise ValueError('uv is unavailable')
    args = [uv, 'run', '--project', str(root), '--locked', 'python', '-m', 'bialy.student_host',
            'reap', '--configuration', str(configuration_path)]
    quoted = ' '.join(json.dumps(arg.replace('%', '%%').replace('$', '$$')) for arg in args)
    unit = ('[Unit]\nDescription=Painted Wolf training resource cleanup\nAfter=network-online.target\n'
            '[Service]\nType=simple\nExecStart=' + quoted + '\nRestart=always\nRestartSec=10\n'
            'User=' + os.environ['USER'] + '\n[Install]\nWantedBy=multi-user.target\n')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'paintedwolf-training-reaper.service'
        path.write_text(unit)
        run(['sudo', 'install', '-m', '644', str(path), '/etc/systemd/system/' + path.name], check=True)
    run(['sudo', 'systemctl', 'daemon-reload'], check=True)
    run(['sudo', 'systemctl', 'enable', '--now', 'paintedwolf-training-reaper.service'], check=True)
    return {'service': 'paintedwolf-training-reaper.service', 'registry': configuration['registry']}
