import json
import os
import shutil
from pathlib import Path

import pytest

from bialy import host_contract
from bialy.cli import main

CHECKOUT = Path(__file__).parent / "fixtures" / "host-checkout"


def test_everything_the_factory_names_of_the_host_is_recorded():
    used = host_contract.check()
    assert "LYCAON_ADDR" in used["environment"] and "PW_TRAINING_CAPTURE_DIR" in used["environment"]
    assert used["commands"]["lycaon-debug decide export"] == {"db", "roots", "out", "retention", "requests"}
    assert used["commands"]["lycaon-debug decide generate"] == {"addr", "tasks", "project", "project-name", "manifest", "unattended", "timeout"}
    assert "LYCAON_API_TOKEN" in used["environment"]
    assert {"lycaon serve", "lycaon diagnostics document-core"} <= set(used["commands"])
    assert used["workflows"]["plan-auto"] == {"extends": "plan", "version": "1.0.1"}


def repository(tmp_path, script="", module="", workflow="plan@1.0.1"):
    (tmp_path / "runner" / "workflows" / "plan-auto").mkdir(parents=True)
    (tmp_path / "src" / "bialy").mkdir(parents=True)
    (tmp_path / "runner" / "run-shard.sh").write_text(script)
    (tmp_path / "src" / "bialy" / "fleet.py").write_text(module)
    (tmp_path / "runner" / "workflows" / "plan-auto" / "workflow.yaml").write_text("id: plan-auto\nextends: %s\n" % workflow)
    return tmp_path


@pytest.mark.parametrize("change,missing", [
    ({"script": "export LYCAON_RETENTION_RECORD=all\n"}, "environment variable LYCAON_RETENTION_RECORD"),
    ({"module": 'env = {"PW_TRAINING_TEACHERS": "1"}\n'}, "environment variable PW_TRAINING_TEACHERS"),
    ({"script": "/opt/bialy/bin/lycaon-debug decide export --db /cfg/store.db \\\n  --candidates /shard/c.jsonl\n"},
     "flag lycaon-debug decide export --candidates"),
    ({"script": 'command=(/opt/bialy/bin/lycaon-debug decide generate --tasks "$(cat /shard/name)"\n  --retries 2)\n'},
     "flag lycaon-debug decide generate --retries"),
    ({"script": "/opt/bialy/bin/lycaon-debug decide replay --db x\n"}, "command lycaon-debug decide replay"),
    ({"module": "subprocess.run(['/opt/bialy/bin/lycaon', 'diagnostics', 'engine'])\n"}, "command lycaon diagnostics engine"),
    ({"module": 'if row.get("schema") == "pw-decide-request/1":\n    pass\n'}, "schema pw-decide-request/1"),
    ({"workflow": "plan@1.0.0"}, "workflow plan-auto extends plan@1.0.0, and the record serves 1.0.1 live"),
])
def test_a_name_the_record_lacks_fails_the_check(tmp_path, change, missing):
    host_contract.check(repository(tmp_path))
    with pytest.raises(host_contract.HostContractError, match="host_contract_unrecorded") as refused:
        host_contract.check(repository(tmp_path / "changed", **change))
    assert refused.value.missing == [missing]


def test_a_shell_invocation_ends_where_its_command_does():
    script = ('(exec env LYCAON_ADDR=x /opt/bialy/bin/lycaon serve) >log 2>&1 &\n'
              'curl --silent http://127.0.0.1:8850/health\n'
              'command=(/opt/bialy/bin/lycaon-debug decide generate --addr u --project "$(cat /shard/name)"\n'
              '  --tasks t)\n'
              'command=(timeout --signal=TERM "${command[@]}")\n')
    assert host_contract.shell_commands(script) == {"lycaon serve": set(), "lycaon-debug decide generate": {"addr", "project", "tasks"}}


def test_arguments_refuse_an_unrecorded_command_or_flag():
    assert host_contract.arguments("lycaon-debug", "decide export", db="store.db", out="rows", roots=None) == \
        ["decide", "export", "--db", "store.db", "--out", "rows"]
    assert host_contract.arguments("lycaon-debug", "decide generate", project_name="flask") == ["decide", "generate", "--project-name", "flask"]
    with pytest.raises(host_contract.HostContractError, match="host_flag_unrecorded"):
        host_contract.arguments("lycaon-debug", "decide export", candidates="rows")
    with pytest.raises(host_contract.HostContractError, match="host_command_unrecorded"):
        host_contract.arguments("lycaon-debug", "decide replay")


def test_a_host_script_is_run_only_with_its_recorded_flags(tmp_path):
    pipe = "scripts/bialy/custom_model/"
    assert host_contract.script("/host", pipe + "export.py", ["--model", "/m", "--out", tmp_path, "--resume"]) == [
        "/host/" + pipe + "export.py", "--model", "/m", "--out", str(tmp_path), "--resume"]
    with pytest.raises(host_contract.HostContractError, match="host_flag_unrecorded") as refused:
        host_contract.script("/host", pipe + "gguf_export.py", ["--merge", "/m", "--loader", "/bin/llama-completion"])
    assert refused.value.missing == [pipe + "gguf_export.py --loader"]
    with pytest.raises(host_contract.HostContractError, match="host_script_unrecorded"):
        host_contract.script("/host", pipe + "prepare.py", [])


def test_every_student_trainer_flag_is_recorded():
    from bialy.student_host.specification import arguments, load, students
    for student in students():
        for options in ({}, {"pilot": True}, {"resume": True}, {"dry_run": True}):
            host_contract.script("/host", "scripts/bialy/custom_model/train.py",
                                 arguments(load(), student, "/models", "/splits", "/run", "cuda:0", "/mirror", **options))


def test_a_missing_host_script_fails_verification(tmp_path):
    checkout = tmp_path / "host"
    shutil.copytree(CHECKOUT, checkout)
    (checkout / "scripts/bialy/custom_model/base_checkpoint.py").unlink()
    with pytest.raises(host_contract.HostContractError, match="host_contract_drift") as refused:
        host_contract.verify(checkout)
    assert refused.value.missing == ["script scripts/bialy/custom_model/base_checkpoint.py"]


def test_the_record_holds_against_the_host_checkout():
    record = host_contract.verify(CHECKOUT)
    declared = host_contract.declared(CHECKOUT)
    assert declared["workflows"]["plan"] == "1.0.1" and "LYCAON_RETENTION_RECORD" not in declared["environment"]
    assert set(record["commands"]["lycaon-debug decide export"]) <= declared["flag_sets"]["lycaon-debug decide export"]


def edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize("path,old,new,missing", [
    ("lycaon/internal/runtime/environment.go", '"PW_TRAINING_EPOCH"', '"PW_CAPTURE_EPOCH"', "environment variable PW_TRAINING_EPOCH"),
    ("lycaon/cmd/lycaon-debug/decide.go", 'fs.String("retention"', 'fs.String("observations"', "flag lycaon-debug decide export --retention"),
    ("lycaon/cmd/lycaon-debug/eval_training.go", '"session"', '"root-session"', "flag lycaon-debug eval training-bundle --session"),
    ("lycaon/cmd/lycaon-debug/decide.go", '"decide corpus"', '"decide catalog"', "command lycaon-debug decide corpus"),
    ("lycaon/cmd/lycaon/main.go", '"document-core"', '"documents"', "command lycaon diagnostics document-core"),
    ("lycaon/cmd/lycaon-debug/decide.go", '"pw-request-row/1"', '"pw-request-row/2"', "schema pw-request-row/1"),
    ("lycaon/config/packs/painted-wolf/plan/workflows/plan/workflow.yaml", "version: 1.0.1", "version: 1.0.2",
     "workflow plan@1.0.1 (the checkout serves 1.0.2 live)"),
    ("scripts/bialy/custom_model/train.py", '"--accumulation"', '"--gradient-accumulation"',
     "flag scripts/bialy/custom_model/train.py --accumulation"),
    ("scripts/bialy/custom_model/gguf_export.py", '"server"', '"loader"', "flag scripts/bialy/custom_model/gguf_export.py --server"),
    ("scripts/bialy/custom_model/serving_check.py", '"--revision"', '"--commit"', "flag scripts/bialy/custom_model/serving_check.py --revision"),
    ("scripts/generation/convert_mlx.py", '"--source-sha256"', '"--pin"', "flag scripts/generation/convert_mlx.py --source-sha256"),
])
def test_a_checkout_that_stopped_declaring_a_recorded_name_fails_verification(tmp_path, path, old, new, missing):
    checkout = tmp_path / "host"
    shutil.copytree(CHECKOUT, checkout)
    edit(checkout / path, old, new)
    with pytest.raises(host_contract.HostContractError, match="host_contract_drift") as refused:
        host_contract.verify(checkout)
    assert refused.value.missing == [missing]


def test_a_sealed_workflow_version_is_not_served_live():
    record = host_contract.load()
    record["workflows"]["security-survey"] = "2.0.0"
    with pytest.raises(host_contract.HostContractError) as refused:
        host_contract.verify(CHECKOUT, record)
    assert refused.value.missing == ["workflow security-survey@2.0.0 (the checkout serves 2.0.1 live)"]


def test_a_directory_that_is_not_a_host_checkout_is_refused(tmp_path):
    with pytest.raises(host_contract.HostContractError, match="host_checkout_unreadable"):
        host_contract.verify(tmp_path)


def test_check_holds_the_record_to_a_given_checkout(tmp_path, capsys):
    assert main(["check", "--host-checkout", str(CHECKOUT)]) == 0
    checkout = tmp_path / "host"
    shutil.copytree(CHECKOUT, checkout)
    edit(checkout / "lycaon/config/packs/painted-wolf/security-survey/workflows/security-survey/workflow.yaml", "2.0.1", "2.1.0")
    assert main(["check", "--host-checkout", str(checkout)]) == 1
    assert "workflow security-survey@2.0.1 (the checkout serves 2.1.0 live)" in capsys.readouterr().err


@pytest.mark.skipif(not os.environ.get("BIALY_HOST_CHECKOUT"), reason="BIALY_HOST_CHECKOUT names no Painted Wolf Code checkout")
def test_the_record_holds_against_a_real_host_checkout():
    host_contract.verify(os.environ["BIALY_HOST_CHECKOUT"])


def test_the_record_names_only_what_the_factory_uses():
    record, used = host_contract.load(), host_contract.uses()
    assert set(record["environment"]) == used["environment"] and set(record["schemas"]) == used["schemas"]
    assert set(record["workflows"]) == {pin["extends"] for pin in used["workflows"].values()}
    assert json.loads(host_contract.RECORD.read_text())["schema"] == host_contract.SCHEMA
