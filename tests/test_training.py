import json
from pathlib import Path

from bialy import training


def test_accelerator_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(training.platform, "system", lambda: "Linux")
    monkeypatch.setattr(training.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(training.shutil, "which", lambda name: None)
    monkeypatch.setattr(training.Path, "exists", lambda self: str(self) == "/dev/kfd")
    assert training.accelerator() == "rocm"
    assert training.torch_index({}) == training.TORCH_INDEX["rocm"]
    assert training.torch_index({"torch_index": "https://x"}) == "https://x"
    monkeypatch.setattr(training.shutil, "which", lambda name: "/usr/bin/" + name if name == "nvidia-smi" else None)
    assert training.accelerator() == "cuda" and training.torch_index({}) is None
    monkeypatch.setattr(training.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(training.platform, "machine", lambda: "arm64")
    assert training.accelerator() == "mps"


def test_environment_maps_rocm_to_torchs_cuda_device(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "accelerator", lambda: "rocm")
    env = training.environment(tmp_path, {"device": "auto"})
    assert env["LYCAON_DECIDE_DEVICE"] == "cuda" and env["HF_HUB_OFFLINE"] == "1"
    assert training.environment(tmp_path, {"device": "cpu"})["LYCAON_DECIDE_DEVICE"] == "cpu"


def test_epoch_cap_replaces_or_adds_the_flag():
    assert training.with_epochs(["--epochs", "45", "--lr", "1"], 2) == ["--epochs", "2", "--lr", "1"]
    assert training.with_epochs(["--lr", "1"], 3) == ["--lr", "1", "--epochs", "3"]
    assert training.with_epochs(["--epochs", "45"], None) == ["--epochs", "45"]


def test_skillreq_eval_keeps_the_writers_evaluation_families(tmp_path):
    rows = [{"meta": {"split": "train", "family": "a"}}, {"meta": {"split": "eval", "family": "b"}}]
    (tmp_path / "judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = training.skillreq_eval(tmp_path / "judged.jsonl", tmp_path / "eval.jsonl")
    assert [json.loads(line)["meta"]["family"] for line in out.read_text().splitlines()] == ["b"]


def test_recipes_without_their_inputs_are_skipped_not_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "venv_python", lambda root, settings: "/py")
    monkeypatch.setattr(training, "backbone", lambda root: tmp_path)
    inputs = {"corpus": "c", "independent_corpus": "i", "train": "t", "val": "v", "skillreq_train": None, "dumps": None,
              "retention_train": tmp_path / "missing.jsonl", "retention_val": tmp_path / "missing.jsonl"}
    assert "skipped" in training.train("E4", inputs, tmp_path, tmp_path, {}, tmp_path / "heads", "l", 10)
    assert "skipped" in training.train("code-rank", inputs, tmp_path, tmp_path, {}, tmp_path / "heads", "l", 10)
    assert "skipped" in training.train("R1", inputs, tmp_path, tmp_path, {}, tmp_path / "heads", "l", 10)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    inputs["retention_train"] = inputs["retention_val"] = empty
    assert "skipped" in training.train("R1", inputs, tmp_path, tmp_path, {}, tmp_path / "heads", "l", 10)


def test_retention_recipe_trains_every_search_point_and_resumes(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(training, "venv_python", lambda root, settings: "/py")
    monkeypatch.setattr(training, "backbone", lambda root: tmp_path)

    def fake_run(args, cwd=None, env=None, timeout=None, log=None):
        seen.append(args)
        Path(args[args.index("--out") + 1]).write_bytes(b"h")
    monkeypatch.setattr(training, "run", fake_run)
    rows = tmp_path / "retention-train.jsonl"
    rows.write_text(json.dumps({"schema": "pw-retention-row/1"}) + "\n")
    inputs = {"corpus": "c", "independent_corpus": "i", "train": "t", "val": "v", "retention_train": rows, "retention_val": rows}
    settings = {"epochs": 2, "retention": {"search": {"lr": [1e-3, 5e-4], "pos_weight": [6.0], "truth": ["needed"]}}}
    out = training.train("R1", inputs, "/co", tmp_path, settings, tmp_path / "heads", "lbl", 99)
    assert sorted(out["search"]) == ["lr0.0005-pw6-needed", "lr0.001-pw6-needed"] and len(seen) == 2
    args = seen[0]
    assert args[:2] == ["/py", "scripts/bialy/train.py"] and args[args.index("--families") + 1] == "retention"
    assert args[args.index("--lr") + 1] == "0.001" and args[args.index("--pos-weight") + 1] == "6.0" and args[args.index("--retention-truth") + 1] == "needed"
    assert args[args.index("--epochs") + 1] == "2" and args[args.index("--out") + 1].endswith("search/retention-lr0.001-pw6-needed.safetensors")
    assert "--label" in args and args[args.index("--label") + 1] == "lbl-lr0.001-pw6-needed"
    # A resumed run keeps the points already trained.
    again = training.train("R1", inputs, "/co", tmp_path, settings, tmp_path / "heads", "lbl", 99)
    assert len(seen) == 2 and all(v.get("reused") for v in again["search"].values())


def test_turn_recipe_arguments(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(training, "venv_python", lambda root, settings: "/py")
    monkeypatch.setattr(training, "backbone", lambda root: tmp_path)

    def fake_run(args, cwd=None, env=None, timeout=None, log=None):
        seen.update(args=args, cwd=cwd, timeout=timeout)
        (tmp_path / "heads/turn-load.safetensors").write_bytes(b"h")
    monkeypatch.setattr(training, "run", fake_run)
    inputs = {"corpus": "c", "independent_corpus": "i", "train": "t", "val": "v"}
    out = training.train("B5", inputs, "/co", tmp_path, {"epochs": 1}, tmp_path / "heads", "lbl", 99)
    assert out["head"].endswith("turn-load.safetensors") and seen["cwd"] == "/co" and seen["timeout"] == 99
    args = seen["args"]
    assert args[:2] == ["/py", "scripts/bialy/train.py"] and "--families" in args and args[args.index("--families") + 1] == "tools"
    assert args[args.index("--epochs") + 1] == "1" and args[args.index("--corpus") + 1] == "i"
