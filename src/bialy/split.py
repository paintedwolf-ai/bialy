"""The run's ancestry, and judged rows split by it.

Before anything is generated, every task the run will drive and every skill-request
family it will write becomes a node of one ancestry graph (campaign.ancestry): a task
joins its prompt group, its scenario, and the repository it works in; a family joins
its repository. Each connected component goes to one partition, a held-out workspace's
to system_test. Rows are then split by the partition of the task they came from, so a
prompt driven under two models, its worker legs, its follow-ups and every row derived
from it land on one side.
"""

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from .campaign import ancestry
from .campaign.journal import CampaignFault


def permission(workspace):
    """A repository's admission receipt: the pinned commit and the license it was admitted under."""
    return hashlib.sha256(("%s\n%s\n%s" % (workspace.name, workspace.commit, workspace.spdx)).encode()).hexdigest()


def held(factory, workspace):
    return "system_test" if workspace.name in factory.holdout() else None


def source(workspace):
    if workspace.kind == "repository":
        return {"source_class": "permitted_repository", "permission_sha256": permission(workspace)}
    return {"source_class": "synthetic"}


def workspace_node(graph, factory, workspace):
    """The node a workspace's descendants join: a repository is shared ancestry; a
    greenfield stack starts empty, so its tasks share nothing through it."""
    if workspace.kind != "repository":
        return None
    return graph.add("repository:" + workspace.name, "repository", holdout=held(factory, workspace), **source(workspace))


def graph(factory, tasks, families=()):
    """The ancestry graph of a run's tasks and skill-request families, each family given
    as (node, workspace name)."""
    g = ancestry.Graph()
    for task in tasks:
        meta = task["meta"]
        workspace = factory.workspace(meta["workspace"])
        node = g.add(ancestry.task_node(task["id"]), "task", holdout=held(factory, workspace),
                     hashes={"prompt": hashlib.sha256(task["prompt"].encode()).hexdigest()}, **source(workspace))
        g.link(node, g.add("group:" + meta["prompt_group"], "prompt_group"))
        if meta.get("scenario"):
            g.link(node, g.add("scenario:" + meta["scenario"], "scenario"))
        repository = workspace_node(g, factory, workspace)
        if repository:
            g.link(node, repository)
    for family, workspace_name in families:
        workspace = factory.workspace(workspace_name)
        node = g.add(family, "skill_request_family", holdout=held(factory, workspace), **source(workspace))
        repository = workspace_node(g, factory, workspace)
        if repository:
            g.link(node, repository)
    return g


def assign(factory, tasks, families=()):
    """The run's frozen allocation of its tasks and families."""
    return ancestry.allocate(graph(factory, tasks, families), factory.split["seed"], factory.split["fractions"])


def read_tasks(tasks_dir):
    out = []
    for path in sorted(Path(tasks_dir).glob("*.jsonl")):
        out.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return out


def split(assignment, rows_paths, out_dir):
    """Every row written to the file of its task's partition (ancestry.FILES). A row from a
    task outside the assignment is refused, and nothing is published until every row is
    placed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {p: 0 for p in ancestry.PARTITIONS}
    components = {p: set() for p in ancestry.PARTITIONS}
    with TemporaryDirectory(prefix=".split-", dir=out_dir) as staging:
        staging = Path(staging)
        with ExitStack() as stack:
            files = {p: stack.enter_context(open(staging / name, "w", encoding="utf-8")) for p, name in ancestry.FILES.items()}
            for path in rows_paths:
                with open(path, encoding="utf-8") as rows:
                    for line in rows:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        task_id = (row.get("meta") or {}).get("task_id")
                        if not task_id:
                            raise CampaignFault("row_without_task", details={"receipt": row.get("receipt")})
                        node = ancestry.task_node(task_id)
                        partition = assignment.partition(node)
                        files[partition].write(line if line.endswith("\n") else line + "\n")
                        counts[partition] += 1
                        components[partition].add(assignment.component(node))
        for name in ancestry.FILES.values():
            (staging / name).replace(out_dir / name)
    return {p: {"rows": counts[p], "components": len(components[p])} for p in ancestry.PARTITIONS}
