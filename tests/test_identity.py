"""Tracked files never name a deployment: no home directories, logins on real
hosts, or private addresses. Deployment details live in git-ignored *.local.*
files. A developer may list their own host and user names, one per line, in
config/identity.local.txt (git ignores it) to have them refused too."""
import ipaddress
import re
import subprocess

import pytest

from bialy import config

HOME = re.compile(r"/(?:Users|home)/(?!user\b)[a-z][\w.-]*")
LOGIN = re.compile(r"\b[a-z_][\w.-]*@(?![\w.-]*\.example\b)(?!github\.com\b)(?!sha\d+:)([a-z0-9][\w-]*(?:\.[\w-]+)*):")
PRIVATE_IP = re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b(?!/)")


def tracked_text():
    names = subprocess.run(["git", "ls-files", "-z"], cwd=config.ROOT, check=True, capture_output=True).stdout.split(b"\0")
    for name in filter(None, names):
        path = config.ROOT / name.decode()
        if path.suffix in {".lock", ".gz", ".png", ".safetensors"} or not path.is_file():
            continue
        try:
            yield name.decode(), path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def local_names():
    path = config.ROOT / "config" / "identity.local.txt"
    if not path.exists():
        return []
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


@pytest.mark.parametrize("pattern", [HOME, LOGIN], ids=["home", "login"])
def test_tracked_files_name_no_deployment(pattern):
    found = [(name, m.group(0)) for name, text in tracked_text() for m in pattern.finditer(text)]
    assert not found


def test_private_addresses_belong_to_the_factory_network(factory):
    bridge = ipaddress.ip_network(factory.network["subnet"])
    found = [(name, m.group(0)) for name, text in tracked_text() for m in PRIVATE_IP.finditer(text)
             if ipaddress.ip_address(m.group(0)) not in bridge]
    assert not found


def test_tracked_files_omit_locally_listed_names():
    names = local_names()
    found = [(path, n) for path, text in tracked_text() for n in names if n in text.lower()]
    assert not found
