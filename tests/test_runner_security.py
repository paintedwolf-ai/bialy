import json
import shutil

import pytest

from bialy import runner_security


def test_nested_policy_preserves_default_deny_and_adds_only_required_calls():
    _, body, apparmor = runner_security.render()
    base = json.loads((runner_security.SOURCE / 'docker-seccomp.json').read_text())
    rendered = json.loads(body)
    assert rendered['defaultAction'] == 'SCMP_ACT_ERRNO'
    assert rendered['syscalls'][:-1] == base['syscalls']
    assert rendered['syscalls'][-1] == {'names': list(runner_security.NESTED_SYSCALLS), 'action': 'SCMP_ACT_ALLOW'}
    assert b'BIALY_PROFILE' not in apparmor


def test_changed_upstream_policy_is_rejected(tmp_path):
    shutil.copytree(runner_security.SOURCE, tmp_path / 'source')
    (tmp_path / 'source/docker-seccomp.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        runner_security.render(tmp_path / 'source')


def test_security_install_is_content_bound_and_requires_apparmor(tmp_path):
    enabled = tmp_path / 'enabled'
    enabled.write_text('N')
    calls = []
    with pytest.raises(ValueError, match='requires AppArmor'):
        runner_security.arguments(tmp_path, enabled=enabled)
    enabled.write_text('Y\n')
    def process(args, **kwargs):
        calls.append(args)
    args = runner_security.arguments(tmp_path, process=process, enabled=enabled)
    assert 'systempaths=unconfined' in args
    assert calls[0][:2] == ['apparmor_parser', '-r']
    assert runner_security.arguments(tmp_path, process=process, enabled=enabled) == args
    path = next((tmp_path / 'security').glob('*/seccomp.json'))
    path.write_text('{}')
    with pytest.raises(ValueError, match='profile changed'):
        runner_security.arguments(tmp_path, process=process, enabled=enabled)
