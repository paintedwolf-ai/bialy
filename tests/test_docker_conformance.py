import json
from pathlib import Path
from types import SimpleNamespace

from host_bundles import contract, write_episode

from bialy import host_bundle

SURFACE = {'tools': [{'name': 'read', 'source': 'standing'}, {'name': 'edit', 'source': 'predicted'}],
           'units': [], 'turn_load_receipt_id': 7, 'engine': 'turn-load', 'catalog_revision': 'r1'}


class Host:
    """One host program reached through Docker or directly. It acts only on the arguments it
    receives, so any difference in the evidence comes from how the factory invoked it."""

    def __init__(self):
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == 'docker':
            mounts = {}
            for flag, spec in zip(command, command[1:], strict=False):
                if flag == '-v':
                    source, target, _ = spec.split(':')
                    mounts[target] = source
            arguments = [next((mounts[target] + value[len(target):] for target in mounts
                               if value == target or value.startswith(target + '/')), value)
                         for value in command[command.index('eval') + 1:]]
        else:
            arguments = command[2:]
        out = Path(arguments[arguments.index('--out') + 1])
        if arguments[0] == 'training-contract':
            out.mkdir()
            (out / 'contract.json').write_text(json.dumps(contract()))
        else:
            session = arguments[arguments.index('--session') + 1]
            capture = {'schema': 'pw-training-turn/1', 'session_id': session, 'message_id': 'm1', 'prompt_surface': SURFACE}
            write_episode(out, sessions=[{'id': session, 'messages': []}], captures=[capture])
        return SimpleNamespace(returncode=0, stdout='', stderr='')


def evidence(root, runner, host):
    _, contract_sha256 = host_bundle.contract(runner, root / 'contract', 'c' * 40, host)
    (root / 'shard').mkdir()
    episode = host_bundle.export(runner, root / 'shard', 'root', root / 'bundle', process=host)
    return contract_sha256, episode


def test_docker_and_an_ordinary_host_produce_the_same_contract_and_prompt_surface_receipts(tmp_path):
    host = Host()
    docker_contract, docker_episode = evidence(tmp_path / 'docker', 'runner:off', host)
    local_contract, local_episode = evidence(tmp_path / 'local', host_bundle.Local('/usr/local/bin/lycaon-debug'), host)
    assert docker_contract == local_contract
    assert docker_episode.manifest_sha256 == local_episode.manifest_sha256
    assert docker_episode.captures == local_episode.captures and docker_episode.captures[0]['prompt_surface'] == SURFACE
    docker_command, local_command = host.commands[1], host.commands[3]
    assert docker_command[:2] == ['docker', 'run'] and local_command[:2] == ['/usr/local/bin/lycaon-debug', 'eval']
    assert docker_command[docker_command.index('eval') + 1] == local_command[2] == 'training-bundle'


def test_the_ordinary_host_reads_each_mount_target_as_its_source():
    command = host_bundle.Local('lycaon-debug').command(
        [('/data/shard', '/capture', 'ro'), ('/data/out', '/out', 'rw')],
        ['training-bundle', '--capture', '/capture', '--captures', '/capture/training-captures', '--out', '/out/bundle',
         '--session', 'root'])
    assert command == ['lycaon-debug', 'eval', 'training-bundle', '--capture', str(Path('/data/shard').resolve()),
                       '--captures', str(Path('/data/shard').resolve()) + '/training-captures',
                       '--out', str(Path('/data/out').resolve()) + '/bundle', '--session', 'root']
