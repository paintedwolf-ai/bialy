"""Committed release anchors pin checksums and training holdouts independently
of the release folder, so rewriting that folder cannot change verification.
"""

import json
from pathlib import Path

from .provenance import file_sha256

ANCHORS = Path(__file__).resolve().parents[2] / "releases"
SCHEMA = "pw-decide-anchor/1"


class AnchorError(ValueError):
    pass


def path(kind, version):
    return ANCHORS / ("%s-%s.json" % (kind, version))


def record(kind, version, release_dir, holdout=None):
    """Each release version has one immutable checksum and holdout anchor."""
    doc = {"schema": SCHEMA, "kind": kind, "version": version, "sha256sums": file_sha256(Path(release_dir) / "SHA256SUMS")}
    if holdout is not None:
        doc["holdout"] = sorted(holdout)
    target = path(kind, version)
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != doc:
        raise AnchorError("%s already anchors a different %s %s; release it under a new version" % (target.name, kind, version))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return target


def match(kind, release_dir):
    """The anchor whose recorded digest matches this release's SHA256SUMS."""
    digest = file_sha256(Path(release_dir) / "SHA256SUMS")
    for candidate in sorted(ANCHORS.glob("%s-*.json" % kind)):
        doc = json.loads(candidate.read_text(encoding="utf-8"))
        if doc.get("schema") == SCHEMA and doc.get("kind") == kind and doc.get("sha256sums") == digest:
            return doc
    raise AnchorError("no %s anchor in releases/ matches this release's SHA256SUMS: it is not a release this "
                      "repository recorded, or its files were changed and checksummed again" % kind)
