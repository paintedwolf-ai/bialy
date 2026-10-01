import json

from bialy import llm, skillreq


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


def test_families_split_whole_and_rows_carry_no_behaviour(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "Chat", Writer)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    out = tmp_path / "rows.jsonl"
    report = skillreq.write(factory, CORPUS, out, "glm-5.3-flash", {"clear": 3, "nearmiss": 1, "multi": 0}, 2, 5, eval_fraction=0.5, workers=2)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert report["failed"] == 0 and rows
    assert all("work-with-containers" not in r["state"]["user"] for r in rows)
    assert all(r["labels"]["skills"] == [] and r["labels"]["tools"] == [] and not r["partial"] for r in rows)
    splits = {}
    for r in rows:
        splits.setdefault(r["meta"]["family"], set()).add(r["meta"]["split"])
    assert all(len(s) == 1 for s in splits.values())


def test_an_acceptance_set_takes_one_split_and_real_candidates(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "Chat", Writer)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    offered = {"floor": ["read"], "loadable": ["edit", "command"], "guides": []}
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
    skillreq.write(factory, CORPUS, out, "deepseek-v4.1-flash", {"clear": 1, "nearmiss": 0, "multi": 0}, 0, 5, workspaces=("react-vite",), workers=1)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert seen and all(u.startswith("The developer works in a new react-vite project (React with TypeScript") for u in seen)
    assert rows and all(r["project"] == "react-vite" and r["meta"]["workspace_kind"] == "stack" for r in rows)
