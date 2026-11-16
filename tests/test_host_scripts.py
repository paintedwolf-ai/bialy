import json
import os
from pathlib import Path

import pytest

from bialy import host_scripts

# The contract of the host commit this factory was last checked against, as that
# commit's scripts/bialy/factory-contract.json states it.
RECORDED = json.loads((Path(__file__).parent / "fixtures" / "factory-contract.json").read_text())


def test_every_flag_the_factory_passes_is_one_the_host_accepts():
    assert host_scripts.missing(RECORDED) == []


def test_a_flag_the_host_dropped_is_reported(tmp_path):
    dropped = json.loads(json.dumps(RECORDED))
    dropped["scripts"]["replay_eval.py"].remove("--tool-truth")
    del dropped["scripts"]["preload.py"]
    assert host_scripts.missing(dropped) == ["preload.py", "replay_eval.py --tool-truth"]
    path = tmp_path / host_scripts.CONTRACT
    assert host_scripts.verify(tmp_path) == ["run.lycaon_checkout lacks " + host_scripts.CONTRACT]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dropped))
    assert "replay_eval.py --tool-truth" in host_scripts.verify(tmp_path)[0]
    path.write_text(json.dumps(dict(RECORDED, schema="pw-factory-cli/0")))
    assert "pw-factory-cli/0" in host_scripts.verify(tmp_path)[0]
    path.write_text(json.dumps(RECORDED))
    assert host_scripts.verify(tmp_path) == []


def test_a_command_cannot_carry_an_undeclared_flag():
    assert host_scripts.command("/py", "replay_eval.py", ["--corpus", Path("c.json"), "--json", "out.json"]) == [
        "/py", "scripts/bialy/replay_eval.py", "--corpus", "c.json", "--json", "out.json"]
    for script, flag in (("replay_eval.py", "--holdout-pack"), ("calibrate.py", "--dump"), ("train.py", "--guides")):
        with pytest.raises(host_scripts.ContractError, match=flag):
            host_scripts.command("/py", script, ["--corpus", "c", flag, "x"])


@pytest.mark.skipif(not os.environ.get("BIALY_HOST_CHECKOUT"), reason="BIALY_HOST_CHECKOUT names no Painted Wolf Code checkout")
def test_the_recorded_contract_is_the_given_checkouts():
    checkout = Path(os.environ["BIALY_HOST_CHECKOUT"])
    assert json.loads((checkout / host_scripts.CONTRACT).read_text()) == RECORDED
    assert host_scripts.verify(checkout) == []
