"""Every command line the factory runs from the pinned checkout's scripts/bialy/.

The host states the flags each of those scripts accepts in
scripts/bialy/factory-contract.json. FLAGS declares the ones the factory passes;
`command` builds a command line from declared flags only, and `verify` refuses a
checkout whose contract lacks any of them, so a flag the host dropped stops a run
at its check stage instead of in the middle of training.
"""
import json
from pathlib import Path

CONTRACT = "scripts/bialy/factory-contract.json"
SCHEMA = "pw-factory-cli/1"
FLAGS = {
    "train.py": ("--corpus", "--train", "--val", "--families", "--tool-truth", "--tool-weight", "--tool-negatives", "--lr",
                 "--pos-weight", "--seed", "--batch-size", "--epochs", "--patience", "--skill-scored", "--skill-zeros",
                 "--rank-levels", "--label", "--out"),
    "rerank/train_rerank.py": ("--dump", "--val-dump", "--metrics", "--out", "--label", "--seed", "--epochs"),
    "calibrate.py": ("--corpus", "--examples", "--engine"),
    "replay_eval.py": ("--corpus", "--examples", "--engine", "--tool-truth", "--json"),
    "parity_probe.py": ("--model", "--head", "--examples", "--engine", "--corpus", "--n", "--tolerance"),
    "preload.py": ("--corpus", "--model"),
}


class ContractError(ValueError):
    pass


def command(python, script, arguments):
    """[python, the script's path in the checkout, *arguments], refused when an argument
    is a flag FLAGS does not declare for that script."""
    declared = FLAGS[script]
    undeclared = sorted({str(a) for a in arguments if str(a).startswith("--")} - set(declared))
    if undeclared:
        raise ContractError("%s is passed %s, which host_scripts.FLAGS does not declare" % (script, ", ".join(undeclared)))
    return [str(python), "scripts/bialy/" + script, *(str(a) for a in arguments)]


def missing(contract):
    """Each declared flag a host contract does not accept, as 'script flag'."""
    if contract.get("schema") != SCHEMA:
        raise ContractError("the host's factory contract is %r, not %s" % (contract.get("schema"), SCHEMA))
    scripts = contract.get("scripts") or {}
    out = []
    for script, flags in sorted(FLAGS.items()):
        if script not in scripts:
            out.append(script)
            continue
        out += ["%s %s" % (script, flag) for flag in flags if flag not in scripts[script]]
    return out


def verify(checkout):
    """Faults that keep this factory from calling the checkout's scripts; empty when it can."""
    path = Path(checkout) / CONTRACT
    if not path.is_file():
        return ["run.lycaon_checkout lacks %s" % CONTRACT]
    try:
        gaps = missing(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, ContractError) as exc:
        return [str(exc)]
    return ["the pinned checkout's scripts no longer accept: %s" % "; ".join(gaps)] if gaps else []
