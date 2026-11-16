"""What the factory asks of the host's binaries, checked against a host checkout.

`config/host-contract.json` records every environment variable, subcommand, flag,
row schema and workflow version the runner and the fleet rely on, and every flag the
student path passes the host's trainer, export and conversion scripts. `check` holds this
repository to the record: the names its sources pass must be recorded. `verify` holds
the record to a host checkout: that checkout's sources must declare each one. A name
the host stopped reading, or never read, therefore fails before a container starts.
"""
import json
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "config" / "host-contract.json"
SCHEMA = "bialy-host-contract/1"
BINARIES = "/opt/bialy/bin/"

# Row and receipt shapes this factory defines itself; every other pw-*/N id in its
# sources is one the host emits and the record names.
FACTORY_SCHEMAS = frozenset((
    "pw-campaign-plan/1", "pw-campaign-run/1", "pw-campaign-source-preflight/2", "pw-campaign-waiting/1",
    "pw-custom-run/2", "pw-data-report/1", "pw-decide-anchor/1", "pw-episode-verdict/2", "pw-evaluation-policy/1",
    "pw-human-review-decision/1", "pw-load-probe/1", "pw-remote-training/1", "pw-resource-lease/2", "pw-review-finished/1",
    "pw-review-item/1", "pw-review-queue/1", "pw-runner-preflight/1", "pw-selector-holdout/1", "pw-selector-policy/1",
    "pw-student-artifact/3", "pw-student-export/2", "pw-student-job/1", "pw-student-release/1", "pw-student-requirements/1",
    "pw-student-source/1",
    "pw-students/2", "pw-teacher-policy/1", "pw-tier-decision/1", "pw-training-campaign-result/2",
    "pw-training-campaign-status/1", "pw-training-rows/1"))

ENVIRONMENT = re.compile(r"\b(?:LYCAON|PW)_[A-Z][A-Z0-9_]*\b")
QUOTED_ENVIRONMENT = re.compile(r"""["']((?:LYCAON|PW)_[A-Z][A-Z0-9_]*)["']""")
SCHEMA_ID = re.compile(r"\bpw-[a-z0-9-]+/[0-9]+\b")
FLAG = re.compile(r"(?<![\w-])--([a-z][a-z0-9-]*)")
FLAG_SET = re.compile(r'NewFlagSet\(\s*"([a-z][a-z0-9 -]*)"')
FLAG_DEFINITION = re.compile(r'\.(?:String|Int|Int64|Uint|Uint64|Float64|Bool|Duration)\(\s*"([a-z][a-z0-9-]*)"'
                             r'|\.(?:StringVar|IntVar|Int64Var|UintVar|Uint64Var|Float64Var|BoolVar|DurationVar|Var)\(\s*[^,()]+,\s*"([a-z][a-z0-9-]*)"')
SHELL_COMMAND = re.compile(re.escape(BINARIES) + r"(lycaon(?:-debug)?)((?:[ \t]+[a-z][a-z-]*)+)")
PYTHON_COMMAND = re.compile(r"""["']""" + re.escape(BINARIES) + r"""(lycaon(?:-debug)?)["']((?:\s*,\s*["'][a-z][a-z-]*["'])+)""")
WORD = re.compile(r"[a-z][a-z-]*")


class HostContractError(Exception):
    """The record, this repository and a host checkout disagree; `missing` lists each name."""

    def __init__(self, code, missing):
        super().__init__("%s: %s" % (code, "; ".join(missing)))
        self.code, self.missing = code, list(missing)


def load(path=RECORD):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        raise HostContractError("host_contract_record_unsupported", [str(record.get("schema"))])
    return record


def arguments(binary, command, record=None, **flags):
    """The argument list of one recorded host subcommand: its words, then `--flag value`
    for each keyword whose value is not None. An unrecorded subcommand or flag is refused."""
    record = record or load()
    name = "%s %s" % (binary, command)
    known = record["commands"].get(name)
    if known is None:
        raise HostContractError("host_command_unrecorded", [name])
    out = command.split()
    for key, value in flags.items():
        flag = key.replace("_", "-")
        if flag not in known:
            raise HostContractError("host_flag_unrecorded", ["%s --%s" % (name, flag)])
        if value is not None:
            out += ["--" + flag, str(value)]
    return out


def script(checkout, path, arguments, record=None):
    """The argument list that runs one recorded host script with `arguments`: its path in
    the checkout, then the arguments. An unrecorded script or flag is refused, so every
    flag the factory passes a trainer or export script is one `verify` holds the host to."""
    record = record or load()
    known = record["scripts"].get(path)
    if known is None:
        raise HostContractError("host_script_unrecorded", [path])
    passed = {str(item)[2:] for item in arguments if str(item).startswith("--")}
    if passed - set(known):
        raise HostContractError("host_flag_unrecorded", ["%s --%s" % (path, flag) for flag in sorted(passed - set(known))])
    return [str(Path(checkout) / path), *map(str, arguments)]


def shell_commands(text):
    """Each host invocation in a shell script, with the flags it passes. An invocation
    inside an array assignment runs to the closing parenthesis; any other, to the end of
    its continued line."""
    found = {}
    for match in SHELL_COMMAND.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        in_array = "=(" in text[line_start:match.start()]
        depth, end = 0, match.end()
        while end < len(text):
            char = text[end]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif char == "\n" and not in_array and text[end - 1] != "\\":
                break
            end += 1
        name = "%s %s" % (match.group(1), " ".join(match.group(2).split()))
        found.setdefault(name, set()).update(FLAG.findall(text[match.end():end]))
    return found


def python_commands(text):
    """Each host invocation written as a literal argument list, without flags."""
    return {"%s %s" % (match.group(1), " ".join(WORD.findall(match.group(2)))): set() for match in PYTHON_COMMAND.finditer(text)}


def sources(root):
    root = Path(root)
    runner = [path for path in sorted((root / "runner").iterdir()) if path.is_file()]
    return runner + sorted((root / "src" / "bialy").rglob("*.py"))


def uses(root=ROOT):
    """What this repository's runner scripts and factory sources name of the host."""
    root = Path(root)
    environment, schemas, commands = set(), set(), {}
    for path in sources(root):
        text = path.read_text(encoding="utf-8")
        environment.update(ENVIRONMENT.findall(text))
        schemas.update(SCHEMA_ID.findall(text))
        scan = python_commands if path.suffix == ".py" else shell_commands
        for name, flags in scan(text).items():
            commands.setdefault(name, set()).update(flags)
    workflows = {}
    for path in sorted((root / "runner" / "workflows").glob("*/workflow.yaml")):
        parent, _, version = str(yaml.safe_load(path.read_text(encoding="utf-8")).get("extends") or "").partition("@")
        workflows[path.parent.name] = {"extends": parent, "version": version}
    return {"environment": environment, "schemas": schemas - FACTORY_SCHEMAS, "commands": commands, "workflows": workflows}


def check(root=ROOT, record=None):
    """Hold this repository to the record."""
    record, used = record or load(), uses(root)
    missing = ["environment variable %s" % name for name in sorted(used["environment"] - set(record["environment"]))]
    missing += ["schema %s" % name for name in sorted(used["schemas"] - set(record["schemas"]))]
    for name, flags in sorted(used["commands"].items()):
        if name not in record["commands"]:
            missing.append("command %s" % name)
            continue
        missing += ["flag %s --%s" % (name, flag) for flag in sorted(flags - set(record["commands"][name]))]
    for name, pin in sorted(used["workflows"].items()):
        if record["workflows"].get(pin["extends"]) != pin["version"]:
            missing.append("workflow %s extends %s@%s, and the record serves %s live" % (
                name, pin["extends"], pin["version"], record["workflows"].get(pin["extends"])))
    if missing:
        raise HostContractError("host_contract_unrecorded", missing)
    return used


SKIPPED = frozenset(("vendor", "testdata", "test", "node_modules", "target"))


def host_files(checkout):
    """The host's own non-test sources: its Go packages and the trainer scripts."""
    checkout = Path(checkout)
    for top, suffixes in ((checkout / "lycaon", (".go",)), (checkout / "scripts" / "bialy", (".py", ".json"))):
        for directory, names, files in os.walk(top):
            names[:] = sorted(name for name in names if name not in SKIPPED)
            for name in sorted(files):
                if name.endswith(suffixes) and not name.endswith("_test.go") and not name.startswith("test_"):
                    yield Path(directory) / name


def declared(checkout):
    """What a host checkout declares: the environment variables and schema ids its sources
    name, each binary's flag sets and command words, and every workflow version served live."""
    checkout = Path(checkout)
    environment, schemas, flag_sets, words = set(), set(), {}, {}
    for path in host_files(checkout):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "LYCAON_" in text or "PW_" in text:
            environment.update(QUOTED_ENVIRONMENT.findall(text))
        if "pw-" in text:
            schemas.update(SCHEMA_ID.findall(text))
        parts = path.relative_to(checkout).parts
        if parts[:2] != ("lycaon", "cmd") or len(parts) < 4:
            continue
        binary = parts[2]
        words.setdefault(binary, set()).update(re.findall(r'"([a-z][a-z-]*)"', text))
        for body in text.split("\nfunc "):
            for name in FLAG_SET.findall(body):
                flag_sets.setdefault("%s %s" % (binary, name), set()).update(a or b for a, b in FLAG_DEFINITION.findall(body))
    workflows = {}
    for path in sorted((checkout / "lycaon" / "config" / "packs").glob("*/*/workflows/*/workflow.yaml")):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        workflows[str(manifest.get("id"))] = str(manifest.get("version"))
    return {"environment": environment, "schemas": schemas, "flag_sets": flag_sets, "words": words, "workflows": workflows}


def verify(checkout, record=None):
    """Hold the record to a host checkout."""
    record, host = record or load(), declared(checkout)
    if not host["words"]:
        raise HostContractError("host_checkout_unreadable", [str(checkout)])
    missing = ["environment variable %s" % name for name in sorted(set(record["environment"]) - host["environment"])]
    missing += ["schema %s" % name for name in sorted(set(record["schemas"]) - host["schemas"])]
    for name, flags in sorted(record["commands"].items()):
        binary, *command = name.split()
        if name in host["flag_sets"]:
            missing += ["flag %s --%s" % (name, flag) for flag in sorted(set(flags) - host["flag_sets"][name])]
        elif flags or not set(command) <= host["words"].get(binary, set()):
            missing.append("command %s" % name)
    for name, version in sorted(record["workflows"].items()):
        if host["workflows"].get(name) != version:
            missing.append("workflow %s@%s (the checkout serves %s live)" % (name, version, host["workflows"].get(name)))
    for path, flags in sorted(record["scripts"].items()):
        source = Path(checkout) / path
        if not source.is_file():
            missing.append("script %s" % path)
            continue
        # A script names a flag in full or, where it declares several in a loop, bare.
        text = source.read_text(encoding="utf-8")
        missing += ["flag %s --%s" % (path, flag) for flag in flags if not re.search(r"""["'](?:--)?%s["']""" % re.escape(flag), text)]
    if missing:
        raise HostContractError("host_contract_drift", missing)
    return record
