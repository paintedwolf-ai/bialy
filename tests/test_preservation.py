import json
import sqlite3

import pytest
from conftest import FakeDocker, sqlite_store

from bialy import preservation


def published(shard):
    return json.loads((shard / preservation.RECORD).read_text())


def test_a_snapshot_of_a_live_wal_store_carries_its_committed_rows(tmp_path):
    live = sqlite_store(tmp_path / "live.db", ["s1"], wal=True)
    with sqlite3.connect(live) as db:
        db.execute("INSERT INTO turns VALUES('s2', 7)")
    assert (tmp_path / "live.db-wal").stat().st_size > 0
    shard = tmp_path / "shard"
    shard.mkdir()
    facts = preservation.container_backup(FakeDocker(live, shard), "c1", shard, "overdue")
    store = shard / preservation.STORE
    assert published(shard) == dict(schema=preservation.SCHEMA, attempt="overdue", **facts)
    assert facts["sha256"] == preservation.digest(store) and store.read_bytes()[18:20] == b"\x01\x01"
    with sqlite3.connect("file:%s?mode=ro" % store, uri=True) as db:
        assert sorted(r for (r,) in db.execute("SELECT root_session FROM turns")) == ["s1", "s2"]
    assert not list(shard.glob("store.*.part")) and not (shard / preservation.FAULT).exists()


@pytest.mark.parametrize("fail", ["backup", "journal", "retention"])
def test_a_failed_backup_is_a_typed_fault_and_never_replaces_a_verified_store(tmp_path, fail):
    live = sqlite_store(tmp_path / "live.db", ["s1"])
    shard = tmp_path / "shard"
    shard.mkdir()
    first = preservation.container_backup(FakeDocker(live, shard), "c1", shard, "completed")
    with pytest.raises(preservation.PreservationFault) as caught:
        preservation.container_backup(FakeDocker(live, shard, fail=fail), "c1", shard, "overdue")
    assert caught.value.code == "store_backup_failed"
    fault = json.loads((shard / preservation.FAULT).read_text())
    assert fault["code"] == "store_backup_failed" and fault["attempt"] == "overdue"
    assert published(shard)["sha256"] == first["sha256"] == preservation.digest(shard / preservation.STORE)


@pytest.mark.parametrize("content, code", [
    (b"not a database at all, just bytes" * 4, "store_unreadable"),
    (b"", "store_unreadable"),
])
def test_an_unreadable_candidate_is_kept_for_diagnosis_beside_the_verified_store(tmp_path, content, code):
    shard = tmp_path / "shard"
    shard.mkdir()
    sqlite_store(shard / "store.completed.part", ["s1"])
    first = preservation.publish(shard, shard / "store.completed.part", "completed")
    candidate = shard / "store.overdue.part"
    candidate.write_bytes(content)
    with pytest.raises(preservation.PreservationFault) as caught:
        preservation.publish(shard, candidate, "overdue")
    assert caught.value.code == code
    assert (shard / "store.overdue.unverified.db").read_bytes() == content and not candidate.exists()
    assert preservation.check(shard)["sha256"] == first["sha256"]


def test_a_store_still_in_wal_mode_or_with_a_side_file_is_not_self_contained(tmp_path):
    wal = sqlite_store(tmp_path / "wal.db", ["s1"], wal=True)
    with pytest.raises(preservation.PreservationFault, match="store_not_self_contained"):
        preservation.verify(wal)
    plain = sqlite_store(tmp_path / "plain.db", ["s1"])
    (tmp_path / "plain.db-journal").write_bytes(b"x")
    with pytest.raises(preservation.PreservationFault, match="store_not_self_contained"):
        preservation.verify(plain)


def test_an_empty_or_truncated_store_does_not_verify(tmp_path):
    empty = tmp_path / "empty.db"
    with sqlite3.connect(empty) as db:
        db.execute("PRAGMA user_version=1")
    with pytest.raises(preservation.PreservationFault, match="store_empty"):
        preservation.verify(empty)
    big = sqlite_store(tmp_path / "big.db", ["s%d" % n for n in range(2000)])
    with sqlite3.connect(big) as db:
        db.execute("CREATE INDEX turns_root ON turns(root_session)")
    data = big.read_bytes()
    big.write_bytes(data[: len(data) // 2])
    with pytest.raises(preservation.PreservationFault) as caught:
        preservation.verify(big)
    assert caught.value.code in {"store_corrupt", "store_unreadable"}


def test_a_retry_after_a_fault_publishes_and_clears_the_fault(tmp_path):
    live = sqlite_store(tmp_path / "live.db", ["s1"])
    shard = tmp_path / "shard"
    shard.mkdir()
    with pytest.raises(preservation.PreservationFault):
        preservation.container_backup(FakeDocker(live, shard, fail="backup"), "c1", shard, "salvage")
    assert preservation.recorded(shard) is None
    preservation.container_backup(FakeDocker(live, shard), "c1", shard, "salvage")
    assert preservation.check(shard)["attempt"] == "salvage" and not (shard / preservation.FAULT).exists()


def test_a_store_that_changed_after_its_record_fails_the_check(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir()
    sqlite_store(shard / preservation.STORE, ["s1"])
    preservation.adopt(shard, "completed")
    with sqlite3.connect(shard / preservation.STORE) as db:
        db.execute("UPDATE turns SET receipt = 9")
    with pytest.raises(preservation.PreservationFault) as caught:
        preservation.check(shard)
    assert caught.value.code in {"store_changed", "store_unrecorded"}


def test_an_interrupted_copy_leaves_no_store_at_the_destination(tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    sqlite_store(source / preservation.STORE, ["s1"])
    value = preservation.adopt(source, "salvage")

    def interrupted(src, dst):
        open(dst, "wb").write(open(src, "rb").read(10))
        raise OSError("interrupted")
    monkeypatch.setattr(preservation.shutil, "copyfile", interrupted)
    with pytest.raises(OSError):
        preservation.copy_verified(source, dest)
    assert not (dest / preservation.STORE).exists() and not (dest / preservation.RECORD).exists()
    monkeypatch.undo()
    assert preservation.copy_verified(source, dest)["sha256"] == value["sha256"]
    assert preservation.check(dest)["sha256"] == preservation.check(source)["sha256"]


def test_each_attempt_keeps_its_own_evidence(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir()
    sqlite_store(shard / preservation.STORE, ["s1"])
    preservation.adopt(shard, "completed")
    (shard / "unpreserved-rows.jsonl").write_text("{}\n")
    first = preservation.archive_attempt(shard, ["unpreserved-rows.jsonl"])
    assert first == shard / "attempts" / "1" and (first / preservation.STORE).exists() and (first / "unpreserved-rows.jsonl").exists()
    assert preservation.archive_attempt(shard, []) is None
    sqlite_store(shard / preservation.STORE, ["s2"])
    assert preservation.archive_attempt(shard, []) == shard / "attempts" / "2"
    assert preservation.digest(first / preservation.STORE) != preservation.digest(shard / "attempts" / "2" / preservation.STORE)


def test_a_snapshot_retains_the_model_output_bodies_its_store_names(tmp_path):
    live = sqlite_store(tmp_path / "live.db", ["s1"])
    shard = tmp_path / "shard"
    shard.mkdir()
    docker = FakeDocker(live, shard)
    preservation.container_backup(docker, "c1", shard, "completed")
    assert ("docker", "exec", "c1", "/opt/bialy/retain-model-outputs.sh", preservation.CONTAINER_SHARD) in docker.calls
