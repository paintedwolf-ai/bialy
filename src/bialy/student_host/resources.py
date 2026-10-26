"""Provider resource leases reconciled by an independent cleanup controller."""
import contextlib
import fcntl
import json
import os
import time
import uuid
from pathlib import Path


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

    def intent(self, specification, lease_seconds):
        if lease_seconds < 60:
            raise ResourceError('resource lease must allow controller recovery')
        with self.locked():
            health = self.root / 'reaper.json'
            if not health.exists() or self.clock() - json.loads(health.read_text())['at'] > 60:
                raise ResourceError('independent reaper is not healthy')
            token = uuid.uuid4().hex
            record = {'schema': 'pw-resource-lease/1', 'token': token, 'specification': specification,
                      'expires_at': self.clock() + lease_seconds, 'state': 'allocating', 'server': None}
            write(self.root / (token + '.json'), record)
            return record

    def allocate(self, record):
        token = record['token']
        with self.locked():
            path = self.root / (token + '.json')
            current = json.loads(path.read_text())
            if current['state'] != 'allocating' or current['expires_at'] <= self.clock():
                raise ResourceError('allocation lease expired or changed')
            servers = self.provider.find(token)
            if len(servers) > 1:
                raise ResourceError('multiple resources match allocation identity')
            server = servers[0] if servers else self.provider.create(token, current['specification'])
            if token not in server['tags']:
                raise ResourceError('provider returned an unowned resource')
            current.update(state='active', server=server)
            write(path, current)
            return current

    def renew(self, token, lease_seconds):
        with self.locked():
            path = self.root / (token + '.json')
            current = json.loads(path.read_text())
            if current['state'] not in {'allocating', 'active'} or current['expires_at'] <= self.clock():
                raise ResourceError('cannot renew an expired or terminal lease')
            current['expires_at'] = self.clock() + lease_seconds
            write(path, current)

    def release(self, token):
        with self.locked():
            path = self.root / (token + '.json')
            current = json.loads(path.read_text())
            return self._release(path, current)

    def _release(self, path, current):
        if current['state'] == 'released':
            return current
        current['state'] = 'releasing'
        write(path, current)
        try:
            servers = self.provider.find(current['token'])
            for server in servers:
                if current['token'] not in server['tags']:
                    raise ResourceError('cleanup refuses an unowned resource')
                self.provider.delete(server)
            if self.provider.find(current['token']):
                raise ResourceError('provider still reports owned resources')
        except Exception as exc:
            current.update(state='cleanup_unconfirmed', cleanup_error=type(exc).__name__)
            write(path, current)
            raise
        current.update(state='released', released_at=self.clock())
        write(path, current)
        return current

    def reap(self):
        results = []
        with self.locked():
            write(self.root / 'reaper.json', {'at': self.clock(), 'pid': os.getpid()})
            for path in sorted(self.root.glob('*.json')):
                if path.name == 'reaper.json':
                    continue
                record = json.loads(path.read_text())
                if record['state'] != 'released' and (record['expires_at'] <= self.clock() or record['state'] in {'releasing', 'cleanup_unconfirmed'}):
                    try:
                        results.append(self._release(path, record))
                    except Exception:
                        results.append(json.loads(path.read_text()))
        return results
