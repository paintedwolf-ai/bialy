"""Release anchors: what a release must be, recorded outside it.

Everything inside a release folder (its checksums, card, and provenance) can be
rewritten together by whoever holds a copy. An anchor is a file in this
repository's releases/ directory, committed when the release is built, that
records the digest of the release's SHA256SUMS and the repositories held out
of training. `verify` and `audit` check a release against its anchor, so a
mirrored or re-uploaded copy that was edited and re-checksummed fails, and the
leakage check reads the holdout set from the anchor, not from the release.
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
    """Write the anchor for a release just built. A version's anchor never changes: a
    rebuild that differs needs a new version."""
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
