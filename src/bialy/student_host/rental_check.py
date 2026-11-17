"""Prove the rental path of a machine configuration on a real machine, without training.

Every mode rents one machine, confirms that its dead-man switch can end the rental, and
then ends the rental a different way:

  release  measure the machine with the job's own preflight, then release it.
  deadman  stop the heartbeats and watch the machine end its own rental.
  reaper   stop renewing the lease and watch the reaper end the rental.
"""
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from . import machines
from .jobs import JobError
from .progress import Stages, rental, seconds
from .reaper import provider
from .remote_training import Remote, failure
from .resources import SETTLE_SECONDS, ResourceError, Resources, write

MODES = ('release', 'deadman', 'reaper')
# The preflight measures and reports; a rental check holds the machine to no tier.
NOTHING = {'disk_gib': 0, 'memory_gib': 0, 'gpu_memory_gib': 0}
# The last deadline a `deadman` check leaves the machine, instead of a job's fifteen minutes.
DEADLINE_SECONDS = 120
# How long past its deadline a rental may take to disappear from the provider's listing.
GRACE_SECONDS = 600
POLL_SECONDS = 15


def ended(account, token, bound, sleep):
    """Whether the provider stopped listing the leased instance within `bound` seconds."""
    for _ in range(bound // POLL_SECONDS):
        sleep(POLL_SECONDS)
        try:
            if not account.owned(token):
                return True
        except (ResourceError, ConnectionError, subprocess.TimeoutExpired):
            pass
    return False


def check(configuration_path, out, mode, *, sleep=time.sleep, clock=time.time, remote_type=Remote):
    configuration = json.loads(Path(configuration_path).read_text())
    machines.validate(configuration)
    provision = configuration.get('provisioning')
    if provision is None or mode not in MODES:
        raise JobError('a rental check takes a machine configuration that rents and one of: ' + ', '.join(MODES))
    account = provider(provision)
    try:
        account.check()
    except ResourceError as exc:
        raise JobError(str(exc)) from exc
    directory = Path(out) / ('rental-check-%s-%s' % (mode, uuid.uuid4().hex))
    directory.mkdir(parents=True)
    descriptor = os.open(directory / 'configuration.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(configuration, stream)
    stages, resources = Stages(directory, clock), Resources(provision['registry'], account)
    result = {'mode': mode, 'passed': False, 'directory': str(directory)}
    proved = False
    try:
        with machines.connect(configuration, directory, remote_type, sleep, stage=stages.enter, clock=clock) as (remote, supervisor):
            token = json.loads((directory / 'resource.json').read_text())['token']
            result['switch'] = json.loads(remote.command([machines.DEADMAN, 'extend', str(machines.HEARTBEAT_SECONDS)], timeout=30))
            stages.enter(mode if mode != 'release' else 'preflight')
            if mode == 'release':
                result['machine'] = machines.inspect(remote, configuration, NOTHING, directory)
                supervisor.check()
                proved = not result['machine']['problems']
            elif mode == 'deadman':
                supervisor.quiet()
                remote.command([machines.DEADMAN, 'extend', str(DEADLINE_SECONDS)], timeout=30)
                proved = ended(account, token, DEADLINE_SECONDS + GRACE_SECONDS, sleep)
                # The lease was renewed throughout, so the reaper had no part in it.
                result['ended_by'] = 'the machine' if proved else None
            else:
                supervisor.abandon()
                proved = ended(account, token, machines.LEASE_SECONDS + GRACE_SECONDS, sleep)
                taken = json.loads(resources.path(token).read_text())['state'] != 'active'
                proved, result['ended_by'] = proved and taken, 'the reaper' if proved and taken else None
            stages.enter('release')
    except Exception as exc:
        result['failure'] = failure(exc)
    stages.close()
    status = json.loads((directory / 'cleanup.json').read_text())['status']
    # A machine that destroyed itself was never acknowledged gone; the ledger doubts an
    # empty listing until SETTLE_SECONDS after the creation. Nothing is rented meanwhile.
    for _ in range(SETTLE_SECONDS // 60 + 1):
        if status == 'confirmed' or not proved:
            break
        sleep(60)
        status = machines.release(configuration, directory, sleep, clock)
    try:
        left = resources.unsettled()
    except (ResourceError, ConnectionError, subprocess.TimeoutExpired) as exc:
        left = {'unavailable': str(exc)}
    result.update(cleanup=status, left=left, stages=seconds(stages.entries(), clock()),
                  rental=rental(machines.Rental(directory).attempts(), clock()),
                  passed=proved and status == 'confirmed' and left == {'leases': [], 'instances': []})
    write(directory / 'rental-check.json', result)
    return result
