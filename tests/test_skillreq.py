import json

from bialy import llm, skillreq, split


class Writer:
    hosted = True

    def __init__(self, base_url, model, **kw):
        self.model = model

    def json(self, system, user, **kw):
        if "Not this" in user:
            return {"requests": ["add a favicon to the landing page", "short"]}
        return {"requests": ["package the service so it runs the same everywhere", "use work-with-containers to ship this"]}


CORPUS = {"catalog_revision": "rev", "skills": [
    {"name": "work-with-containers", "description": "Build and run container images.", "card": "Skill: work-with-containers"},
    {"name": "craft-icons-and-chrome", "description": "Design favicons and app icons.", "card": "Skill: craft-icons-and-chrome"}]}


def test_a_request_that_names_its_skill_is_dropped():
    assert skillreq.names_skill("please use Work With Containers here", "work-with-containers")
    assert skillreq.names_skill("work-with-containers", "work-with-containers")
    assert not skillreq.names_skill("run it in a container", "work-with-containers")


def ancestry_for(factory, families, none_families):
    return split.assign(factory, [], skillreq.family_nodes(factory, CORPUS, families, none_families, 7))


def test_families_take_their_ancestry_partition_and_rows_carry_no_behaviour(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "Chat", Writer)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    out = tmp_path / "rows.jsonl"
    families = {"clear": 3, "nearmiss": 1, "multi": 0}
    assignment = ancestry_for(factory, families, 2)
    report = skillreq.write(factory, CORPUS, out, "glm-5.3-flash", families, 2, 5, workers=2, assignment=assignment)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert report["failed"] == 0 and rows and sum(report["splits"].values()) == len(rows)
    assert all("work-with-containers" not in r["state"]["user"] for r in rows)
    assert all(r["labels"]["skills"] == [] and r["labels"]["tools"] == [] and not r["partial"] for r in rows)
    assert all(r["meta"]["split"] == assignment.partition(skillreq.family_node(r["meta"]["family"])) for r in rows)
    held = [r for r in rows if r["project"] in factory.holdout()]
    assert all(r["meta"]["split"] == "system_test" for r in held)


def test_a_writer_needs_an_ancestry_or_an_explicit_split(factory, tmp_path):
    import pytest

    with pytest.raises(ValueError, match="ancestry assignment or one explicit split"):
        skillreq.write(factory, CORPUS, tmp_path / "rows.jsonl", "glm-5.3-flash", {"clear": 1, "nearmiss": 0, "multi": 0}, 0, 5)


def test_an_acceptance_set_takes_one_split_and_real_candidates(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "Chat", Writer)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    offered = {"floor": ["read"], "loadable": ["edit", "command"]}
    out = tmp_path / "rows.jsonl"
    skillreq.write(factory, CORPUS, out, "deepseek-v4.1-flash", {"clear": 1, "nearmiss": 0, "multi": 0}, 0, 5, split="accept", offered=offered,
                   workspaces=("zod",), workers=1)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all(r["meta"]["split"] == "accept" and r["offered"] == offered and r["project"] == "zod" for r in rows)


def test_a_family_can_be_set_in_a_new_project_on_a_stack(factory, tmp_path, monkeypatch):
    seen = []

    class Recorder(Writer):
        def json(self, system, user, **kw):
            seen.append(user)
            return super().json(system, user, **kw)

    monkeypatch.setattr(llm, "Chat", Recorder)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    out = tmp_path / "rows.jsonl"
    families = {"clear": 1, "nearmiss": 0, "multi": 0}
    assignment = split.assign(factory, [], skillreq.family_nodes(factory, CORPUS, families, 0, 7, workspaces=("react-vite",)))
    skillreq.write(factory, CORPUS, out, "deepseek-v4.1-flash", families, 0, 5, workspaces=("react-vite",), workers=1, assignment=assignment)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert seen and all(u.startswith("The developer works in a new react-vite project (React with TypeScript") for u in seen)
    assert rows and all(r["project"] == "react-vite" and r["meta"]["workspace_kind"] == "stack" for r in rows)
