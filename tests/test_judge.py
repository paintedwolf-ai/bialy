import json
import re

import pytest

from bialy import judge, llm


class FakeChat:
    """Scores a candidate by whether its text mentions a word the request contains."""

    hosted = False

    def __init__(self, scores=None, drop=0):
        self.scores, self.drop, self.calls = scores or {}, drop, []

    def json(self, system, user, **kw):
        self.calls.append(kw)
        request, listing = user.split("\n\nCandidates:\n\n")
        out = {}
        for entry in listing.split("\n\n"):
            number, text = entry.split(". ", 1)
            out[number] = self.scores.get(text, 4 if set(re.findall(r"\w+", text.lower())) & set(re.findall(r"\w+", request.lower())) else 0)
        for number in list(out)[: self.drop]:
            del out[number]
        return {"scores": out}


def test_scores_map_back_to_names_whatever_the_order():
    cards = {"commit": "Commit changes", "deploy": "Deploy the app", "lint": "Lint the code"}
    for seed in range(5):
        assert judge.score_cards(FakeChat(), "please commit my work", cards, seed) == {"commit": 4, "deploy": 0, "lint": 0}


def test_decoding_is_greedy_seeded_and_schema_constrained():
    chat = FakeChat()
    judge.score_cards(chat, "x", {"a": "A"}, 11)
    call = chat.calls[0]
    assert call["temperature"] == 0.0 and call["seed"] == 11 and call["thinking"] is False
    assert call["schema"]["properties"]["scores"]["required"] == ["1"]


def test_an_incomplete_verdict_is_refused():
    cards = {name: name for name in "abcdefghij"}
    with pytest.raises(ValueError):
        judge.score_cards(FakeChat(drop=2), "x", cards, 1)


def test_agreement_records_pair_the_same_cards():
    a = {"session": "s", "receipt": 1, "judge": {"model": "m1"}, "labels": {"skill_scores": {"x": 3, "y": 0}, "requests": [{"scores": {"edit": 4}}]}}
    b = {"session": "s", "receipt": 1, "judge": {"model": "m2"}, "labels": {"skill_scores": {"x": 2, "y": 0}, "requests": [{"scores": {"edit": 4}}]}}
    records = judge.agreement_records(a, b)
    assert [(r["unit"], r["candidate"], r["first_score"], r["second_score"]) for r in records] == [
        ("skills", "x", 3, 2), ("skills", "y", 0, 0), ("need0", "edit", 4, 4)]


def test_weighted_kappa():
    assert judge.weighted_kappa([(0, 0), (4, 4), (2, 2), (1, 1)]) == 1.0
    assert judge.weighted_kappa([(0, 4), (4, 0), (0, 4), (4, 0)]) < 0
    assert judge.weighted_kappa([]) is None


def test_run_writes_judged_rows_and_the_second_judge_sample(factory, tmp_path, monkeypatch):
    from conftest import row

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: FakeChat())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(row("flask", "g%d" % n, n, requests=[{"need": "edit a file", "exact": [], "after": []}])) + "\n" for n in range(6)))
    corpus = {"skills": [{"name": "commit", "card": "Commit changes"}], "tools": [{"name": "edit", "card": "Edit a file"}, {"name": "command", "card": "Run it"}]}
    report = judge.run(factory, corpus, rows, tmp_path / "out.jsonl", workers=2, second=1.0)
    judged = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert report["rows"] == 6 and report["failed"] == 0
    assert all(r["judge"]["model"] == factory.judge_for(r["model"]).id for r in judged)
    assert judged[0]["labels"]["requests"][0]["scores"] == {"edit": 4, "command": 0}
    assert judged[0]["labels"]["tool_scores"] == {"edit": 0, "command": 0}
    assert judged[0]["judge"]["units"] == ["skills", "tools", "requests"]
    second = judged[0]["labels"]["second_scores"]
    assert second["model"] != judged[0]["judge"]["model"] and second["tools"] == {"edit": 0, "command": 0} and second["requests"] == [{"edit": 4, "command": 0}]
    records = [json.loads(line) for line in (tmp_path / "out.jsonl.agreement.jsonl").read_text().splitlines()]
    assert len(records) == report["second_judge_pairs"] == 6 * 5 and report["weighted_kappa"] == 1.0


def test_a_pass_on_one_unit_keeps_earlier_labels_and_samples_only_that_unit(factory, tmp_path, monkeypatch):
    from conftest import row

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: FakeChat())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    rows = tmp_path / "rows.jsonl"
    earlier = {"model": "glm-5.3-flash", "units": ["skills", "requests"]}
    rows.write_text("".join(json.dumps(dict(row("flask", "g%d" % n, n, skill_scores={"commit": 3}), state={"host": "coordinator", "user": "edit something"},
                                            judge=earlier)) + "\n" for n in range(4)))
    corpus = {"skills": [{"name": "commit", "card": "Commit changes"}], "tools": [{"name": "edit", "card": "Edit a file"}, {"name": "command", "card": "Run it"}]}
    report = judge.run(factory, corpus, rows, tmp_path / "out.jsonl", workers=2, second=1.0, units=("tools",))
    judged = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert all(r["labels"]["skill_scores"] == {"commit": 3} and r["labels"]["tool_scores"] == {"edit": 4, "command": 0} for r in judged)
    assert judged[0]["judge"]["units"] == ["skills", "tools", "requests"]
    records = [json.loads(line) for line in (tmp_path / "out.jsonl.agreement.jsonl").read_text().splitlines()]
    assert {r["unit"] for r in records} == {"tools"} and report["second_judge_pairs"] == 4 * 2


def test_a_row_judged_by_another_model_is_refused(factory, tmp_path, monkeypatch):
    from conftest import row

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: FakeChat())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row("flask", "g", 1, judge_model="deepseek-v4.1-flash")) + "\n")
    corpus = {"skills": [], "tools": [{"name": "edit", "card": "Edit a file"}]}
    report = judge.run(factory, corpus, rows, tmp_path / "out.jsonl", workers=1, second=0.0, units=("tools",))
    assert report["rows"] == 0 and report["failed"] == 1


def test_candidate_order_is_seeded_per_row_and_unit():
    row = {"session": "s", "receipt": 3}
    assert judge.order_seed(row, "skills") == judge.order_seed(row, "skills")
    assert judge.order_seed(row, "skills") != judge.order_seed(row, "need0")


def test_a_failed_row_is_tried_again(factory, tmp_path, monkeypatch):
    from conftest import row

    class Flaky(FakeChat):
        seen = set()

        def json(self, system, user, **kw):
            if user not in Flaky.seen:
                Flaky.seen.add(user)
                raise RuntimeError("rate limit exceeded")
            return super().json(system, user, **kw)

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: Flaky())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(row("flask", "g%d" % n, n)) + "\n" for n in range(3)))
    corpus = {"skills": [{"name": "commit", "card": "Commit changes"}], "tools": [{"name": "edit", "card": "Edit a file"}]}
    report = judge.run(factory, corpus, rows, tmp_path / "out.jsonl", workers=1, second=0.0)
    assert report["rows"] == 3 and report["failed"] == 0


def test_a_full_pass_replaces_another_judges_labels(factory, tmp_path, monkeypatch):
    from conftest import row

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: FakeChat())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row("flask", "g", 1, judge_model="qwen3.6-35b-a3b", skill_scores={"commit": 4})) + "\n")
    corpus = {"skills": [{"name": "commit", "card": "Commit changes"}], "tools": [{"name": "edit", "card": "Edit a file"}]}
    report = judge.run(factory, corpus, rows, tmp_path / "out.jsonl", workers=1, second=0.0)
    judged = json.loads((tmp_path / "out.jsonl").read_text())
    assert report["rows"] == 1 and judged["judge"]["model"] == "glm-5.3-flash" and judged["labels"]["skill_scores"] == {"commit": 0}


def test_a_need_the_second_judge_fails_stays_unpaired(factory, tmp_path, monkeypatch):
    from conftest import row

    class Refuses(FakeChat):
        def json(self, system, user, **kw):
            if self.model == "deepseek-v4p1-flash" and user.startswith("Request:\nedit a file"):
                raise ValueError("no JSON in reply: ''")
            return super().json(system, user, **kw)

    def chat(base_url, model, **kw):
        c = Refuses()
        c.model = model.rsplit("/", 1)[-1]
        return c

    monkeypatch.setattr(llm, "Chat", chat)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    needs = [{"need": "edit a file", "exact": [], "after": [], "scores": {"edit": 1}}, {"need": "run it", "exact": [], "after": []}]
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row("flask", "g", 1, requests=needs)) + "\n")
    corpus = {"skills": [{"name": "commit", "card": "Commit changes"}], "tools": [{"name": "edit", "card": "Edit a file"}, {"name": "command", "card": "Run it"}]}
    report = judge.run(factory, corpus, rows, tmp_path / "out.jsonl", workers=1, second=1.0)
    second = json.loads((tmp_path / "out.jsonl").read_text())["labels"]["second_scores"]
    assert report["rows"] == 1 and second["missed"] == ["need0"]
    assert second["requests"] == [None, {"edit": 0, "command": 4}] and "tools" in second


def test_a_partial_pass_keeps_earlier_second_scores(factory, tmp_path, monkeypatch):
    from conftest import row

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: FakeChat())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    earlier = {"model": "deepseek-v4.1-flash", "missed": ["need0"], "skills": {"commit": 2}, "tools": {"edit": 1}, "requests": [None]}
    judged = dict(row("flask", "g", 1, skill_scores={"commit": 3}, second_scores=earlier), judge={"model": "glm-5.3-flash", "units": list(judge.UNITS)})
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(judged) + "\n")
    corpus = {"skills": [{"name": "commit", "card": "Commit changes"}], "tools": [{"name": "edit", "card": "Edit a file"}, {"name": "command", "card": "Run it"}]}
    judge.run(factory, corpus, rows, tmp_path / "out.jsonl", workers=1, second=1.0, units=("tools",))
    second = json.loads((tmp_path / "out.jsonl").read_text())["labels"]["second_scores"]
    assert second["skills"] == {"commit": 2} and second["requests"] == [None] and second["missed"] == ["need0"]
    assert second["tools"] == {"edit": 0, "command": 0}


def test_a_stopped_run_resumes_without_judging_finished_rows_again(factory, tmp_path, monkeypatch):
    from conftest import row

    calls = []

    class Counting(FakeChat):
        fail = True

        def json(self, system, user, **kw):
            calls.append(user)
            if Counting.fail and "request 3" in user:
                raise RuntimeError("connection reset")
            return super().json(system, user, **kw)

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: Counting())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(row("flask", "g%d" % n, n)) + "\n" for n in range(5)))
    corpus = {"catalog_revision": "rev", "skills": [{"name": "commit", "card": "Commit changes"}], "tools": [{"name": "edit", "card": "Edit a file"}]}
    out = tmp_path / "out.jsonl"
    first = judge.run(factory, corpus, rows, out, workers=1, second=0.0, rounds=1)
    assert first["rows"] == 4 and first["failed"] == 1
    judged_first = len(calls)
    Counting.fail = False
    second = judge.run(factory, corpus, rows, out, workers=1, second=0.0, rounds=1)
    assert second["rows"] == 5 and second["failed"] == 0 and second["attempts"] == 2
    assert all("request 3" in c for c in calls[judged_first:])
    manifest = json.loads((tmp_path / "out.jsonl.manifest.json").read_text())
    assert manifest["completed"] and manifest["rows"] == 5


def test_a_rerun_with_another_rubric_or_input_is_refused(factory, tmp_path, monkeypatch):
    from conftest import row

    monkeypatch.setattr(llm, "Chat", lambda base_url, model, **kw: FakeChat())
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row("flask", "g", 1)) + "\n")
    corpus = {"catalog_revision": "rev", "skills": [{"name": "commit", "card": "Commit changes"}], "tools": []}
    out = tmp_path / "out.jsonl"
    judge.run(factory, corpus, rows, out, workers=1, second=0.0)
    monkeypatch.setattr(judge, "SKILL_SYSTEM", judge.SKILL_SYSTEM + " A changed rule.")
    with pytest.raises(ValueError, match="different"):
        judge.run(factory, corpus, rows, out, workers=1, second=0.0)


def test_account_refusal_stops_pending_rows_without_accepting_partial_consensus(factory, tmp_path, monkeypatch):
    from conftest import row

    class Suspended(Exception):
        status_code = 412

    calls = []

    class AccountChat(FakeChat):
        def json(self, system, user, **kw):
            calls.append(user)
            if len(calls) == 4:
                raise Suspended('account suspended')
            return super().json(system, user, **kw)

    monkeypatch.setattr(llm, 'Chat', lambda *args, **kw: AccountChat())
    monkeypatch.setenv('FIREWORKS_API_KEY', 'test')
    source, out = tmp_path / 'in.jsonl', tmp_path / 'out.jsonl'
    source.write_text(''.join(json.dumps(row('flask', 'g%d' % n, n)) + '\n' for n in range(5)))
    corpus = {'skills': [{'name': 'commit', 'card': 'Commit changes'}], 'tools': []}
    report = judge.run(factory, corpus, source, out, workers=1, second=1, units=('skills',))
    assert report['rows'] == 1 and report['failed'] == 4
    assert report['fatal_error'] == 'account suspended' and len(calls) == 4
    saved = out.read_text().splitlines()
    report = judge.run(factory, corpus, source, out, workers=1, second=1, units=('skills',))
    assert report['rows'] == 5 and report['failed'] == 0 and report['fatal_error'] is None
    assert len(calls) == 12
    assert saved[0] in out.read_text().splitlines()
