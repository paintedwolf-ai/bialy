"""Vast.ai CLI operations on the on-demand instances labelled with a lease token."""
import json
import os
import re
import subprocess
from pathlib import Path

from .resources import NoOffer, ResourceError

REQUIRED_SPECIFICATION = {'gpu_name', 'image', 'max_price', 'gpu_memory_gib', 'memory_gib', 'disk_gib', 'download_mbps'}
ALLOWED_SPECIFICATION = REQUIRED_SPECIFICATION | {'num_gpus'}
# The provider stores the startup command in a bounded field.
STARTUP_LIMIT = 4000
LABEL = 'pw-training-'
PINS = Path(__file__).resolve().parents[3] / 'train-host/student/pins.env'


def cli_version(path=PINS):
    """The CLI release the controller is set up with. Its listing endpoint and its
    confirmation prompts changed between releases, so no other release is driven."""
    found = re.search(r'^VASTAI_CLI_VERSION=(\S+)$', Path(path).read_text(), re.M)
    if not found:
        raise ResourceError('no Vast.ai CLI version is pinned')
    return found.group(1)


def label(token):
    return LABEL + token


def direct(item):
    """The instance's own address and the host port mapped to its SSH port, or None. The
    provider's proxy address is never used: every artifact would cross it."""
    mapped = ((item.get('ports') or {}).get('22/tcp') or [{}])[0].get('HostPort')
    port = int(mapped) if str(mapped or '').isdigit() else 0
    return {'address': item['public_ipaddr'], 'port': port} if item.get('public_ipaddr') and 0 < port < 65536 else None


def price(item):
    """Dollars an hour for the whole instance, storage included, or None when not stated."""
    value = item.get('dph_total')
    return float(value) if type(value) in (int, float) and value > 0 else None


def normalize(item):
    """An instance in any state. It has an endpoint only once it is running and reachable.
    `host` names the physical machine, which outlives the instance."""
    return {'id': str(item['id']), 'state': str(item.get('actual_status') or 'pending'),
            'interruptible': bool(item.get('is_bid')), 'children': [],
            'host': str(item['machine_id']) if item.get('machine_id') else None, 'price_per_hour': price(item),
            'endpoint': direct(item) if item.get('actual_status') == 'running' else None}


class VastAI:
    def __init__(self, api_key_file=None, run=subprocess.run):
        self.api_key_file, self.run = api_key_file, run

    def call(self, *arguments):
        environment = None
        if self.api_key_file:
            environment = dict(os.environ, VAST_API_KEY=Path(self.api_key_file).read_text().strip())
        result = self.run(['vastai', '--raw', *arguments], capture_output=True, text=True, timeout=120, env=environment,
                          stdin=subprocess.DEVNULL)
        if result.returncode:
            raise ResourceError('Vast.ai operation did not confirm success')
        try:
            return json.loads(result.stdout) if result.stdout.strip() else None
        except ValueError as exc:
            raise ResourceError('Vast.ai returned an unreadable response') from exc

    def check(self):
        """Refuse a CLI other than the pinned release. Nothing is asked of the provider."""
        pinned = cli_version()
        try:
            result = self.run(['vastai', '--version'], capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ResourceError('the Vast.ai CLI is not installed; train-host/student/setup.sh installs vastai==' + pinned) from exc
        found = re.search(r'\d+(\.\d+)+', result.stdout or '')
        if result.returncode or not found or found.group(0) != pinned:
            raise ResourceError('the Vast.ai CLI is %s; this controller drives vastai==%s (train-host/student/pins.env)'
                                % (found.group(0) if found else 'not answering', pinned))
        return pinned

    def instances(self):
        result = self.call('show', 'instances')
        if not isinstance(result, list) or any(not isinstance(item, dict) or not item.get('id') for item in result):
            raise ResourceError('Vast.ai instance list returned an unexpected shape')
        return result

    def owned(self, token):
        return [normalize(item) for item in self.instances() if item.get('label') == label(token)]

    def labelled(self):
        """Every instance of the account that carries a lease label, by its token."""
        found = {}
        for item in self.instances():
            if str(item.get('label') or '').startswith(LABEL):
                found.setdefault(item['label'][len(LABEL):], []).append(normalize(item))
        return found

    def offer(self, specification, avoid=()):
        """The cheapest matching offer that is not on a machine in `avoid`. Nothing is rented."""
        query = ['gpu_name=' + specification['gpu_name'], 'num_gpus=%d' % specification.get('num_gpus', 1),
                 'gpu_ram>=%d' % specification['gpu_memory_gib'], 'cpu_ram>=%d' % specification['memory_gib'],
                 'disk_space>=%d' % specification['disk_gib'], 'inet_down>=%d' % specification['download_mbps'],
                 'dph_total<=%s' % specification['max_price'],
                 'rentable=true', 'verified=true']
        offers = self.call('search', 'offers', '--type', 'on-demand', '--order', 'dph_total', ' '.join(query))
        if not isinstance(offers, list) or any(not isinstance(offer, dict) or not offer.get('id') for offer in offers):
            raise ResourceError('Vast.ai offer search returned an unexpected shape')
        offers = [offer for offer in offers if str(offer.get('machine_id')) not in avoid]
        if not offers:
            raise NoOffer('Vast.ai has no matching on-demand offer')
        return {'id': str(offers[0]['id']), 'host': str(offers[0]['machine_id']) if offers[0].get('machine_id') else None,
                'price_per_hour': price(offers[0])}

    def create(self, token, specification, startup, offer):
        if len(startup.encode()) > STARTUP_LIMIT:
            raise ResourceError('startup command exceeds the provider limit')
        # No bid price is passed: a bid makes the instance interruptible.
        created = self.call('create', 'instance', offer['id'], '--image', specification['image'],
                            '--disk', str(specification['disk_gib']), '--ssh', '--direct', '--label', label(token),
                            '--onstart-cmd', startup)
        if not isinstance(created, dict) or not created.get('new_contract'):
            raise ResourceError('Vast.ai creation returned an invalid contract identity')
        return {'id': str(created['new_contract']), 'state': 'pending', 'interruptible': False, 'children': [],
                'host': offer['host'], 'price_per_hour': offer['price_per_hour'], 'endpoint': None}

    def remaining(self, instance):
        return [str(item['id']) for item in self.instances() if str(item['id']) == instance['id']]

    def destroy(self, instance):
        """True when the provider acknowledged the destruction. The request is made by identity
        whatever the listing says, because a new instance can be missing from it."""
        try:
            # Without --yes the CLI asks for confirmation and exits 0 having destroyed nothing.
            self.call('destroy', 'instance', instance['id'], '--yes')
        except (ResourceError, subprocess.TimeoutExpired):
            if self.remaining(instance):
                raise
            return False
        return True
