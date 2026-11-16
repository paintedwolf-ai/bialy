"""Ownership ledger for rented machines.

A lease is written before the provider is asked for anything and carries the token the
provider resource is labelled with, so whatever exists can be found and destroyed from the
ledger alone. A lease is released only when the provider confirms nothing owned remains.
"""
import contextlib
import fcntl
import json
import os
import re
import time
import uuid
from pathlib import Path

# How long a provider listing that shows nothing is doubted, since a new instance can be
# missing from it: a creation whose outcome is unknown, an instance whose destruction was
# not acknowledged, and a released lease are checked again for this long after they began.
SETTLE_SECONDS = 900


class ResourceError(RuntimeError):
    pass


def write(path, value):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def merged(known, observed, now):
    """One record per instance, keeping children and acknowledgements a later snapshot omits."""
    instances = {}
    for instance in (*known, *observed):
        previous = instances.get(instance['id'], {})
        children = {tuple(child) for child in (*previous.get('children', ()), *instance.get('children', ()))}
        instances[instance['id']] = {**previous, **instance, 'children': sorted(map(list, children)),
                                     'recorded_at': previous.get('recorded_at', instance.get('recorded_at', now))}
    return list(instances.values())


class Resources:
    def __init__(self, root, provider, clock=time.time):
        self.root, self.provider, self.clock = Path(root), provider, clock
        self.root.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def locked(self):
        with (self.root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def path(self, token):
        return self.root / (token + '.json')

    def live(self, record):
        return record['state'] in {'allocating', 'active'} and self.clock() < min(record['expires_at'], record['deadline_at'])

    def intent(self, specification, lease_seconds, lifetime_seconds):
        if lease_seconds < 60 or lifetime_seconds < lease_seconds:
            raise ResourceError('resource lease must allow controller recovery')
        with self.locked():
            health = self.root / 'reaper.json'
            if not health.exists() or self.clock() - json.loads(health.read_text())['at'] > 60:
                raise ResourceError('independent reaper is not healthy')
            token, now = uuid.uuid4().hex, self.clock()
            record = {'schema': 'pw-resource-lease/2', 'token': token, 'specification': specification, 'state': 'allocating',
                      'created_at': now, 'expires_at': now + lease_seconds, 'deadline_at': now + lifetime_seconds,
                      'create_started_at': None, 'instance': None, 'instances': []}
            write(self.path(token), record)
            return record

    def allocate(self, token, startup):
        """Create the leased instance, or adopt the one an earlier attempt created."""
        with self.locked():
            current = json.loads(self.path(token).read_text())
            if current['state'] != 'allocating' or not self.live(current):
                raise ResourceError('allocation lease expired or changed')
            observed = self.provider.owned(token)
            current['instances'] = merged(current['instances'], observed, self.clock())
            write(self.path(token), current)
            if len(current['instances']) > 1:
                raise ResourceError('multiple resources match allocation identity')
            if not current['instances']:
                if current['create_started_at'] is not None:
                    raise ResourceError('allocation outcome is unknown')
                current['create_started_at'] = self.clock()
                write(self.path(token), current)
                created = self.provider.create(token, current['specification'], startup)
                current['instances'] = merged([], [created], self.clock())
            current.update(state='active', instance=current['instances'][0]['id'])
            write(self.path(token), current)
            return current

    def renew(self, token, lease_seconds):
        """Extend a lease the reaper has not taken. The reaper decides under the same lock, so
        a renewal that was only delayed past the expiry still wins if it arrives first."""
        with self.locked():
            current = json.loads(self.path(token).read_text())
            if current['state'] not in {'allocating', 'active'} or self.clock() >= current['deadline_at']:
                raise ResourceError('cannot renew a terminal lease or one past its deadline')
            current['expires_at'] = min(self.clock() + lease_seconds, current['deadline_at'])
            write(self.path(token), current)

    def observe(self, token):
        """The provider's present view of the leased instance, or None when it lists none."""
        with self.locked():
            current = json.loads(self.path(token).read_text())
            if current['state'] != 'active':
                raise ResourceError('resource lease is not active')
            return self._observe(current)

    def _observe(self, current):
        observed = self.provider.owned(current['token'])
        current['instances'] = merged(current['instances'], observed, self.clock())
        write(self.path(current['token']), current)
        if len(current['instances']) > 1:
            raise ResourceError('multiple resources match allocation identity')
        return next((instance for instance in observed if instance['id'] == current['instance']), None)

    def recover(self, token, lease_seconds):
        with self.locked():
            current = json.loads(self.path(token).read_text())
            if current['state'] != 'active' or not self.live(current):
                raise ResourceError('resource lease cannot be recovered')
            if self._observe(current) is None:
                raise ResourceError('owned resource is missing')
            current['expires_at'] = min(self.clock() + lease_seconds, current['deadline_at'])
            write(self.path(token), current)
            return current

    def release(self, token):
        with self.locked():
            return self._release(json.loads(self.path(token).read_text()))

    def _release(self, current):
        path, token, now = self.path(current['token']), current['token'], self.clock()
        if current['state'] != 'released':
            current['state'] = 'releasing'
            write(path, current)
        try:
            # Children are recorded before anything is destroyed: a server that is gone no longer names them.
            current['instances'] = merged(current['instances'], self.provider.owned(token), now)
            write(path, current)
            for instance in current['instances']:
                if not instance.get('destroyed') or self.provider.remaining(instance):
                    instance['destroyed'] = self.provider.destroy(instance)
                    write(path, current)
            if self.provider.owned(token) or any(self.provider.remaining(instance) for instance in current['instances']):
                raise ResourceError('provider still reports owned resources')
            doubted = [instance['recorded_at'] for instance in current['instances'] if not instance['destroyed']]
            if not current['instances'] and current['create_started_at'] is not None:
                doubted.append(current['create_started_at'])
            if any(now < since + SETTLE_SECONDS for since in doubted):
                raise ResourceError('provider lists nothing owned, but has not acknowledged that nothing exists')
        except Exception as exc:
            current.update(state='cleanup_unconfirmed', cleanup_error=type(exc).__name__)
            write(path, current)
            raise
        if current['state'] != 'released':
            current.update(state='released', released_at=now)
            current.pop('cleanup_error', None)
            write(path, current)
        return current

    def due(self, record):
        if record['state'] == 'released':
            return self.clock() < record['released_at'] + SETTLE_SECONDS
        return not self.live(record)

    def reap(self):
        """Release every lease that is expired, past its deadline, or not yet confirmed gone.
        One unreadable or failing lease never keeps the others from being released."""
        results = []
        with self.locked():
            write(self.root / 'reaper.json', {'at': self.clock(), 'pid': os.getpid()})
            for path in sorted(self.root.glob('*.json')):
                if not re.fullmatch('[0-9a-f]{32}', path.stem):
                    continue
                try:
                    try:
                        record = json.loads(path.read_text())
                    except ValueError:
                        # The file name is the ownership token, which is all a release needs.
                        record = {'schema': 'pw-resource-lease/2', 'token': path.stem, 'state': 'cleanup_unconfirmed',
                                  'create_started_at': self.clock(), 'instance': None, 'instances': []}
                    if self.due(record):
                        results.append(self._release(record))
                except Exception as exc:
                    results.append({'token': path.stem, 'state': 'cleanup_unconfirmed', 'cleanup_error': type(exc).__name__})
        return results
