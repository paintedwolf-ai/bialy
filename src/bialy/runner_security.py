"""Container boundaries that permit the application's nested confinement."""
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / 'runner' / 'security'
NESTED_SYSCALLS = ('clone', 'unshare', 'mount', 'umount2', 'open_tree', 'move_mount',
                   'mount_setattr', 'fsopen', 'fsconfig', 'fsmount', 'pidfd_getfd', 'process_vm_readv')


def render(source=SOURCE):
    provenance = json.loads((source / 'provenance.json').read_text())
    raw = (source / 'docker-seccomp.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance['seccomp_sha256']:
        raise ValueError('runner seccomp source checksum mismatch')
    policy = json.loads(raw)
    policy['syscalls'].append({'names': list(NESTED_SYSCALLS), 'action': 'SCMP_ACT_ALLOW'})
    seccomp = json.dumps(policy, sort_keys=True).encode()
    template = (source / 'runner.apparmor').read_text()
    name = 'bialy-runner-' + hashlib.sha256(seccomp + template.encode()).hexdigest()[:20]
    return name, seccomp, template.replace('BIALY_PROFILE', name).encode()


def arguments(root, process=subprocess.run, enabled=Path('/sys/module/apparmor/parameters/enabled')):
    if not enabled.is_file() or enabled.read_text().strip() != 'Y':
        raise ValueError('runner nested confinement requires AppArmor')
    name, seccomp, apparmor = render()
    directory = Path(root) / 'security' / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename, body in (('seccomp.json', seccomp), ('apparmor', apparmor)):
        path = directory / filename
        if path.exists():
            if path.read_bytes() != body:
                raise ValueError('runner security profile changed')
        else:
            with path.open('xb') as stream:
                stream.write(body)
    process(['apparmor_parser', '-r', str(directory / 'apparmor')], check=True, capture_output=True)
    return ['--security-opt', 'apparmor=' + name, '--security-opt', 'seccomp=' + str(directory / 'seccomp.json'),
            '--security-opt', 'systempaths=unconfined']
