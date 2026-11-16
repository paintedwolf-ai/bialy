"""One ancestry graph and the partition every node of it inherits.

Tasks, prompt groups, scenario templates, repositories, forks, variants and generated
families are nodes; declared relationships (a task's prompt group, the repository it
works in, the template it varies) are edges. Each connected component goes to exactly
one of train, validation, development and system_test, so nothing that shares ancestry
can sit on both sides of a split. Sessions, workers, follow-ups, head rows and student
encodings inherit the partition of the task they descend from.

Allocation happens before generation and is deterministic: explicit holdouts first,
then per-stratum minimums, then fractions, all ordered by a hash of the frozen seed and
the component's members. The assignment is written once with the hashes of the graph
and of itself, and every reader refuses one that does not match them.
"""

import hashlib
import json
import os
from pathlib import Path

from .journal import CampaignFault, canonical, digest

PARTITIONS = ("train", "validation", "development", "system_test")
# The order minimums are filled in: the sealed and selection sets first.
FILL_ORDER = ("system_test", "development", "validation", "train")
SOURCE_CLASSES = ("synthetic", "permitted_repository")
HASH_KINDS = ("prompt", "fixture", "solution")
SCHEMA = "bialy-ancestry/1"
FILES = {"train": "train.jsonl", "validation": "val.jsonl", "development": "development.jsonl", "system_test": "holdout.jsonl"}


class Graph:
    """Nodes with their source facts and the edges that join them."""

    def __init__(self):
        self.nodes = {}
        self.edges = set()

    def add(self, node, kind, *, source_class=None, permission_sha256=None, holdout=None, stratum=None, hashes=None):
        """Declare a node. A node declared twice must declare the same facts."""
        if not node or not kind:
            raise CampaignFault("ancestry_node_invalid")
        if source_class is not None and source_class not in SOURCE_CLASSES:
            raise CampaignFault("source_not_permitted", details={"node": node, "source_class": source_class})
        if source_class == "permitted_repository" and not permission_sha256:
            raise CampaignFault("repository_permission_missing", details={"node": node})
        if holdout is not None and holdout not in PARTITIONS:
            raise CampaignFault("ancestry_holdout_invalid", details={"node": node})
        unknown = set(hashes or {}) - set(HASH_KINDS)
        if unknown:
            raise CampaignFault("ancestry_hash_kind_invalid", details={"node": node, "kinds": sorted(unknown)})
        facts = {"kind": kind, "source_class": source_class, "permission_sha256": permission_sha256,
                 "holdout": holdout, "stratum": stratum, "hashes": dict(sorted((hashes or {}).items()))}
        prior = self.nodes.get(node)
        if prior is not None and prior != facts:
            raise CampaignFault("ancestry_node_conflict", details={"node": node})
        self.nodes[node] = facts
        return node

    def link(self, a, b):
        """Join two declared nodes: they share ancestry and therefore a partition."""
        if a not in self.nodes or b not in self.nodes:
            raise CampaignFault("ancestry_edge_unknown_node", details={"edge": sorted((a, b))})
        if a != b:
            self.edges.add(tuple(sorted((a, b))))

    def sha256(self):
        return digest({"nodes": self.nodes, "edges": sorted(self.edges)})

    def components(self):
        parent = {node: node for node in self.nodes}

        def root(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node
        for a, b in sorted(self.edges):
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        groups = {}
        for node in sorted(self.nodes):
            groups.setdefault(root(node), []).append(node)
        return [tuple(members) for members in sorted(groups.values())]


def component_id(members):
    return hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]


def position(seed, cid):
    """A component's place in [0, 1) for this seed."""
    return int(hashlib.sha256(("%s:%s" % (seed, cid)).encode()).hexdigest()[:12], 16) / float(1 << 48)


def check_sources(graph, members):
    """A component is admissible only when it has a declared source and every declared
    source in it is admissible; a forbidden ancestor taints every descendant."""
    classes = {graph.nodes[m]["source_class"] for m in members} - {None}
    if not classes:
        raise CampaignFault("ancestry_source_undeclared", details={"nodes": list(members[:5])})


def check_duplicates(graph, components):
    """Identical prompts, fixtures or solutions in two components would cross a split."""
    seen = {}
    for cid, members in components.items():
        for member in members:
            for kind, value in graph.nodes[member]["hashes"].items():
                other = seen.setdefault((kind, value), cid)
                if other != cid:
                    raise CampaignFault("duplicate_across_components", details={"kind": kind, "components": sorted((other, cid))})


def holdout_of(graph, members):
    declared = {graph.nodes[m]["holdout"] for m in members} - {None}
    if len(declared) > 1:
        raise CampaignFault("holdout_conflict", details={"partitions": sorted(declared), "nodes": list(members[:5])})
    return next(iter(declared), None)


def stratum_of(graph, members):
    declared = {graph.nodes[m]["stratum"] for m in members} - {None}
    if len(declared) > 1:
        raise CampaignFault("stratum_conflict", details={"strata": sorted(declared), "nodes": list(members[:5])})
    return next(iter(declared), None)


def check_fractions(fractions):
    if set(fractions) - set(PARTITIONS) or any(v < 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise CampaignFault("ancestry_fractions_invalid", details={"fractions": fractions})


def by_fraction(seed, cid, fractions):
    u, edge = position(seed, cid), 0.0
    for partition in PARTITIONS:
        edge += fractions.get(partition, 0.0)
        if u < edge:
            return partition
    return next(p for p in reversed(PARTITIONS) if fractions.get(p, 0.0) > 0)


def allocate(graph, seed, fractions, minimums=None):
    """The partition of every component: its explicit holdout; otherwise, within its
    stratum, the per-partition `minimums` filled in hash order, then `fractions`.
    A stratum too small for its minimums is refused with the exact deficit."""
    check_fractions(fractions)
    minimums = minimums or {}
    if set(minimums) - set(PARTITIONS) or any(n < 0 for n in minimums.values()):
        raise CampaignFault("ancestry_minimums_invalid", details={"minimums": minimums})
    components = {component_id(members): members for members in graph.components()}
    for members in components.values():
        check_sources(graph, members)
    check_duplicates(graph, components)
    partition, strata = {}, {}
    for cid, members in components.items():
        held = holdout_of(graph, members)
        if held:
            partition[cid] = held
        else:
            strata.setdefault(stratum_of(graph, members), []).append(cid)
    deficits = {}
    for stratum, cids in sorted(strata.items(), key=lambda kv: str(kv[0])):
        ordered = sorted(cids, key=lambda c: (position(seed, c), c))
        held_here = [partition[c] for c, m in components.items() if c in partition and stratum_of(graph, m) == stratum]
        need = {p: max(0, minimums.get(p, 0) - held_here.count(p)) for p in FILL_ORDER}
        if sum(need.values()) > len(ordered):
            deficits[str(stratum)] = sum(need.values()) - len(ordered)
            continue
        for p in FILL_ORDER:
            for _ in range(need[p]):
                partition[ordered.pop(0)] = p
        for cid in ordered:
            partition[cid] = by_fraction(seed, cid, fractions)
    if deficits:
        raise CampaignFault("coverage_short", details={"deficits": deficits})
    body = {"schema": SCHEMA, "seed": str(seed), "fractions": dict(sorted(fractions.items())),
            "minimums": dict(sorted(minimums.items())), "graph_sha256": graph.sha256(),
            "components": {cid: {"partition": partition[cid], "nodes": list(members)} for cid, members in sorted(components.items())}}
    return Assignment(dict(body, assignment_sha256=digest(body)))


class Assignment:
    """A frozen allocation: the partition of every node of the graph it was made from."""

    def __init__(self, document):
        if document.get("schema") != SCHEMA:
            raise CampaignFault("ancestry_schema_unknown", details={"schema": document.get("schema")})
        body = {k: v for k, v in document.items() if k != "assignment_sha256"}
        if digest(body) != document.get("assignment_sha256"):
            raise CampaignFault("ancestry_assignment_changed")
        self.document = document
        self.node_partition, self.node_component = {}, {}
        for cid, component in document["components"].items():
            for node in component["nodes"]:
                self.node_partition[node] = component["partition"]
                self.node_component[node] = cid

    @property
    def sha256(self):
        return self.document["assignment_sha256"]

    def partition(self, node):
        try:
            return self.node_partition[node]
        except KeyError:
            raise CampaignFault("node_outside_ancestry", details={"node": node}) from None

    def component(self, node):
        try:
            return self.node_component[node]
        except KeyError:
            raise CampaignFault("node_outside_ancestry", details={"node": node}) from None

    def counts(self):
        out = {p: 0 for p in PARTITIONS}
        for component in self.document["components"].values():
            out[component["partition"]] += 1
        return out

    def save(self, path):
        """Write once; an existing assignment must be this one."""
        path = Path(path)
        body = canonical(self.document)
        if path.exists():
            if load(path).sha256 != self.sha256:
                raise CampaignFault("ancestry_assignment_changed", details={"path": path.name})
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_text(body + "\n", encoding="utf-8")
        os.replace(part, path)
        return path


def load(path):
    return Assignment(json.loads(Path(path).read_text(encoding="utf-8")))


def task_node(task_id):
    return "task:" + str(task_id)
