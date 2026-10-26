import dataclasses
import json
import os
import random
import subprocess
from pathlib import Path

import pytest
import yaml

from bialy import config, fleet, tasks


def test_local_workers_never_fall_back_to_hosted(factory):
    hosted = [m for m in factory.models if m.hosted and "generator" in m.roles]
    scoped = dataclasses.replace(factory, models=hosted, run={"local_workers_only": True})
    with pytest.raises(config.ConfigError, match="local generator"):
        fleet.sidecar_files(scoped, {}, 0)
    scoped = dataclasses.replace(factory, run={"local_workers_only": True})
    _, policy, _, _, _ = fleet.sidecar_files(scoped, {}, 0)
    parsed = yaml.safe_load(policy)
    assert parsed["lite"]["provider_id"].startswith("vllm-")
    assert all(m["provider_id"].startswith("vllm-") for m in parsed["agent_pool"]["models"])


@pytest.mark.parametrize("routing", [
    {"archetype_generators": {"review": []}},
    {"archetype_generators": {"review": ["missing"]}},
    {"generator_weights": {"a": 0, "b": 0}},
    {"generator_weights": {"a": float("nan")}},
    {"generator_weights": {"a": -1}},
])
def test_invalid_generator_routing_fails_closed(factory, routing):
    with pytest.raises(config.ConfigError):
        tasks.pick_generator(dataclasses.replace(factory, run=routing), "review", ["a", "b"], random.Random(7))


def test_task_writing_obeys_routing_and_resume_is_reproducible(factory, tmp_path, monkeypatch):
    stack = factory.workspace("react-vite")
    arch = dataclasses.replace(next(a for a in factory.archetypes if a.id == "new_project"), per_workspace=18)
    routing = {"archetype_generators": {"new_project": ["qwen3.6-35b-a3b"]}}
    scoped = dataclasses.replace(factory, repos=[], stacks=[stack], archetypes=[arch], run=routing)

    class Writer:
        def json(self, system, user, seed, **kw):
            return {"tasks": [{"prompt": "Build a small app for the project seed %s and test it." % seed, "follow_ups": []}]}

    monkeypatch.setattr(config.Factory, "chats", lambda self, model: [Writer()])
    original = tmp_path / "original"
    tasks.generate(scoped, tmp_path, original, workers=1)
    raw = original / "raw/react-vite.jsonl"
    entries = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(entries) == 3 and {e["author"] for e in entries} == {"qwen3.6-35b-a3b"}
    resumed = tmp_path / "resumed"
    (resumed / "raw").mkdir(parents=True)
    (resumed / "raw/react-vite.jsonl").write_text(json.dumps(entries[-1]) + "\n")
    tasks.generate(scoped, tmp_path, resumed, workers=1)
    assert (original / "react-vite.jsonl").read_bytes() == (resumed / "react-vite.jsonl").read_bytes()


def test_generated_references_do_not_impersonate_host_attachments():
    prompt = tasks.maybe_reference("Review src/a.py please.", ["src/a.py"], 1.0, random.Random(7))
    assert "existing file `src/a.py`" in prompt
    assert "```attachment" not in prompt and "[User attached" not in prompt


def test_readonly_review_does_not_autoapprove_implementation(factory):
    review = next(a for a in factory.archetypes if a.id == "design_review")
    assert set(review.workflows) <= {"recon-auto"}


def test_empty_recollection_replaces_old_usage_and_retention(tmp_path):
    run = tmp_path / "run"
    (run / "shards").mkdir(parents=True)
    for name in ("requests.jsonl", "retention.jsonl"):
        (run / name).write_text('{"stale": true}\n')
    fleet.collect(run, "engine-off", tmp_path / "tasks")
    assert (run / "requests.jsonl").read_text() == ""
    assert (run / "retention.jsonl").read_text() == ""


def test_provider_keys_are_absent_in_the_sidecar_environment():
    helper = Path(__file__).parents[1] / "runner/provider-credentials.sh"
    env = dict(os.environ, BIALY_HOSTED_PROVIDERS="one:BIALY_TEST_KEY,two:BIALY_TEST_KEY", BIALY_TEST_KEY="secret",
               GITHUB_SEARCH_API_KEY="github-secret", GITLAB_SEARCH_API_KEY="gitlab-secret", STACKEXCHANGE_API_KEY="stack-secret")
    result = subprocess.run(
        ["bash", "-eu", "-c", 'source "$1"; (clear_provider_environment; env); test "$BIALY_TEST_KEY" = secret', "test", str(helper)],
        env=env, capture_output=True, text=True, check=True,
    )
    assert "BIALY_TEST_KEY=" not in result.stdout
    assert not any(key + "=" in result.stdout for key in ("GITHUB_SEARCH_API_KEY", "GITLAB_SEARCH_API_KEY", "STACKEXCHANGE_API_KEY"))


@pytest.mark.parametrize("provider_path", ["providers", "web-research/providers"])
def test_provider_credentials_are_json_encoded_and_never_curl_arguments(tmp_path, provider_path):
    helper = Path(__file__).parents[1] / "runner/provider-credentials.sh"
    curl = tmp_path / "curl"
    curl.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$BIALY_TEST_ARGS"\ncat > "$BIALY_TEST_BODY"\nprintf 200\n')
    curl.chmod(0o755)
    key = 'quoted"key\\with\nnewline'
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], BIALY_TEST_KEY=key,
               BIALY_TEST_ARGS=str(tmp_path / "args"), BIALY_TEST_BODY=str(tmp_path / "body"))
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", 'source "$1"; store_provider_credential provider BIALY_TEST_KEY /tmp/header "$2"',
         "test", str(helper), provider_path],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout == "200"
    assert json.loads((tmp_path / "body").read_text()) == {"api_key": key}
    args = (tmp_path / "args").read_text()
    assert key not in args and "@-\n" in args and "@/tmp/header\n" in args
    assert "http://127.0.0.1:8850/v1/%s/provider/credential" % provider_path in args
