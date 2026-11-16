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


def test_skill_requests_are_read_by_their_familys_partition(tmp_path):
    rows = [{"meta": {"split": p, "family": p[0]}} for p in ("train", "validation", "development", "system_test")]
    (tmp_path / "judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for partition in ("train", "validation"):
        out = training.skillreq_partition(tmp_path / "judged.jsonl", partition, tmp_path / (partition + ".jsonl"))
        assert [json.loads(line)["meta"]["family"] for line in out.read_text().splitlines()] == [partition[0]]


def test_recipes_without_their_inputs_are_skipped_not_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "venv_python", lambda root, settings: "/py")
    monkeypatch.setattr(training, "backbone", lambda root: tmp_path)
    inputs = {"corpus": "c", "train": "t", "val": "v", "skillreq_train": None, "skillreq_eval": None, "dumps_train": [], "dumps_val": []}
    assert "skipped" in training.train("E4", inputs, tmp_path, tmp_path, {}, tmp_path / "heads", "l", 10)
    assert "skipped" in training.train("code-rank", inputs, tmp_path, tmp_path, {}, tmp_path / "heads", "l", 10)
    assert "skipped" in training.train("code-rank", dict(inputs, dumps_train=["a.jsonl"]), tmp_path, tmp_path, {}, tmp_path / "heads", "l", 10)


def recorded(tmp_path, monkeypatch, head):
    seen = {}
    monkeypatch.setattr(training, "venv_python", lambda root, settings: "/py")
    monkeypatch.setattr(training, "backbone", lambda root: tmp_path)

    def fake_run(args, cwd=None, env=None, timeout=None, log=None):
        seen.update(args=args, cwd=cwd, timeout=timeout)
        (tmp_path / "heads" / head).write_bytes(b"h")
    monkeypatch.setattr(training, "run", fake_run)
    return seen


def test_code_rank_trains_on_the_named_train_dumps_and_selects_on_the_validation_dumps(tmp_path, monkeypatch):
    seen = recorded(tmp_path, monkeypatch, "code-rank.safetensors")
    stray = tmp_path / "dumps" / "holdout.jsonl"
    stray.parent.mkdir()
    stray.write_text("{}\n")
    inputs = {"dumps_train": [tmp_path / "dumps/train/a.jsonl", tmp_path / "dumps/train/b.jsonl"], "dumps_val": [tmp_path / "dumps/validation/c.jsonl"]}
    out = training.train("code-rank", inputs, "/co", tmp_path, {}, tmp_path / "heads", "lbl", 99)
    args = seen["args"]
    assert out["head"].endswith("code-rank.safetensors") and args[:2] == ["/py", "/co/scripts/bialy/rerank/train_rerank.py"]
    assert [args[i + 1] for i, a in enumerate(args) if a == "--dump"] == [str(p) for p in inputs["dumps_train"]]
    assert [args[i + 1] for i, a in enumerate(args) if a == "--val-dump"] == [str(p) for p in inputs["dumps_val"]]
    assert args[args.index("--metrics") + 1].endswith("code-rank.metrics.json") and str(stray) not in args


def test_rank_recipe_mixes_training_families_and_validates_on_validation_requests(tmp_path, monkeypatch):
    seen = recorded(tmp_path, monkeypatch, "unit-rank.safetensors")
    session = tmp_path / "train.jsonl"
    session.write_text(json.dumps({"meta": {"task_id": "t1"}}) + "\n")
    families = tmp_path / "skillreq.jsonl"
    families.write_text("".join(json.dumps({"meta": {"source": "skillreq", "split": split, "family": "f%d" % n}}) + "\n"
                                for n in range(40) for split in ("train", "validation")))
    inputs = {"corpus": "c", "train": session, "val": "v", "skillreq_train": families, "skillreq_eval": tmp_path / "eval.jsonl"}
    training.train("E4", inputs, "/co", tmp_path, {}, tmp_path / "heads", "lbl", 99)
    args = seen["args"]
    mixed = [json.loads(line)["meta"] for line in Path(args[args.index("--train") + 1]).read_text().splitlines()]
    assert mixed[0] == {"task_id": "t1"} and all(m["split"] == "train" for m in mixed[1:]) and 5 < len(mixed) - 1 < 35
    assert args[args.index("--val") + 1] == str(tmp_path / "eval.jsonl") and args[args.index("--corpus") + 1] == "c"


def test_turn_recipe_trains_on_the_hosts_exported_corpus(tmp_path, monkeypatch):
    seen = recorded(tmp_path, monkeypatch, "turn-load.safetensors")
    inputs = {"corpus": "c", "train": "t", "val": "v"}
    out = training.train("B5", inputs, "/co", tmp_path, {"epochs": 1}, tmp_path / "heads", "lbl", 99)
    assert out["head"].endswith("turn-load.safetensors") and seen["cwd"] == "/co" and seen["timeout"] == 99
    args = seen["args"]
    assert args[:2] == ["/py", "/co/scripts/bialy/train.py"] and "--families" in args and args[args.index("--families") + 1] == "tools"
    assert args[args.index("--epochs") + 1] == "1" and args[args.index("--corpus") + 1] == "c"
