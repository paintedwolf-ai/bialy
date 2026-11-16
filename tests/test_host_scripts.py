import os
import shutil
from pathlib import Path

import pytest

from bialy import host_scripts

# A reduced host checkout: each script's argument parser as the host declares it.
CHECKOUT = Path(__file__).parent / "fixtures" / "host-checkout"


def test_every_flag_the_factory_passes_is_one_the_host_declares():
    assert host_scripts.verify(CHECKOUT) == []
    assert "--val-dump" in host_scripts.declared(CHECKOUT, "rerank/train_rerank.py")


@pytest.mark.parametrize("script, flag", [("replay_eval.py", "--tool-truth"), ("rerank/partition.py", "--val-dump"), ("train.py", "--tool-negatives")])
def test_a_flag_the_host_dropped_or_renamed_is_reported(tmp_path, script, flag):
    shutil.copytree(CHECKOUT, tmp_path / "co")
    path = tmp_path / "co" / host_scripts.SCRIPTS / script
    path.write_text(path.read_text().replace('"%s"' % flag, '"%s-renamed"' % flag))
    faults = host_scripts.verify(tmp_path / "co")
    assert len(faults) == 1 and flag in faults[0] and "no longer accepts" in faults[0]


def test_a_missing_script_is_reported(tmp_path):
    shutil.copytree(CHECKOUT, tmp_path / "co")
    (tmp_path / "co" / host_scripts.SCRIPTS / "preload.py").unlink()
    assert host_scripts.verify(tmp_path / "co") == ["run.lycaon_checkout lacks scripts/bialy/preload.py"]


def test_a_command_cannot_carry_an_unrecorded_flag():
    assert host_scripts.command("/py", "replay_eval.py", ["--corpus", Path("c.json"), "--json", "out.json"]) == [
        "/py", "scripts/bialy/replay_eval.py", "--corpus", "c.json", "--json", "out.json"]
    for script, flag in (("replay_eval.py", "--holdout-pack"), ("calibrate.py", "--dump"), ("train.py", "--guides")):
        with pytest.raises(host_scripts.ContractError, match=flag):
            host_scripts.command("/py", script, ["--corpus", "c", flag, "x"])


@pytest.mark.skipif(not os.environ.get("BIALY_HOST_CHECKOUT"), reason="BIALY_HOST_CHECKOUT names no Painted Wolf Code checkout")
def test_the_given_checkout_declares_every_recorded_flag():
    assert host_scripts.verify(os.environ["BIALY_HOST_CHECKOUT"]) == []
