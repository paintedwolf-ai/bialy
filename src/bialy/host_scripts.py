"""Every command line the factory runs from the pinned checkout's scripts/bialy/.

FLAGS records the flags the factory passes each script. `command` builds a command
line from recorded flags only, and `verify` holds the record to a host checkout: each
script's argument parser must declare every recorded flag, so a flag the host dropped
or renamed stops a run at its check stage instead of in the middle of training.
"""
import re
from pathlib import Path

SCRIPTS = "scripts/bialy"
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
ARGUMENT = re.compile(r'add_argument\(\s*"(--[a-z][a-z0-9-]*)"')
IMPORT = re.compile(r"^(?:import|from)\s+(\w+)", re.M)


class ContractError(ValueError):
    pass


def command(python, script, arguments):
    """[python, the script's path in the checkout, *arguments], refused when an argument
    is a flag FLAGS does not record for that script."""
    unrecorded = sorted({str(a) for a in arguments if str(a).startswith("--")} - set(FLAGS[script]))
    if unrecorded:
        raise ContractError("%s is passed %s, which host_scripts.FLAGS does not record" % (script, ", ".join(unrecorded)))
    return [str(python), "%s/%s" % (SCRIPTS, script), *(str(a) for a in arguments)]


def declared(checkout, script):
    """The flags a host script's parser declares: its own add_argument calls and those of
    the modules beside it that it imports, which may declare arguments for it. None when
    the checkout has no such script."""
    path = Path(checkout) / SCRIPTS / script
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    flags = set(ARGUMENT.findall(text))
    for module in IMPORT.findall(text):
        beside = path.with_name(module + ".py")
        if beside.is_file():
            flags.update(ARGUMENT.findall(beside.read_text(encoding="utf-8")))
    return flags


def verify(checkout):
    """Faults that keep this factory from calling the checkout's scripts; empty when it can."""
    faults = []
    for script, flags in sorted(FLAGS.items()):
        found = declared(checkout, script)
        if found is None:
            faults.append("run.lycaon_checkout lacks %s/%s" % (SCRIPTS, script))
            continue
        missing = [flag for flag in flags if flag not in found]
        if missing:
            faults.append("%s/%s no longer accepts %s" % (SCRIPTS, script, ", ".join(missing)))
    return faults
