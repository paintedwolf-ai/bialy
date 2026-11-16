# Training host

What the factory needs on the machine that trains.

## Decision heads

Heads are trained, calibrated, qualified and released by the selector
(`src/bialy/campaign/selector.py`), through `bialy run` or a campaign. It calls
the pinned Painted Wolf Code checkout's `scripts/bialy/` trainers, calibration,
replay and parity probe; nothing here trains a head by hand. See
[`docs/running.md`](../docs/running.md).

`requirements.txt` pins the trainer's packages. The selector makes the trainer
environment from it with `uv`, under `<root>/venvs/train`, with the torch wheel
index chosen from the accelerator found. `laya` is pinned here and nowhere else;
the pin is a version whose API the pinned checkout's `scripts/bialy/train.py`
uses.

## Student

The student host trains and exports every student declared in
`config/students.json` (E4B, 12B, and 26B-A4B) using the pinned Painted Wolf
Code trainer. That file is the only student list: it declares each model
revision and LoRA targets, and one context policy for pilots, full training,
and serving: 65,536 tokens, thinking disabled, and no token truncation. Adding
a student is a new entry there; collection, datasets, training, export, and
evaluation follow it. Student-specific dataset directories
contain `train.jsonl`, `val.jsonl`, `holdout.jsonl`, and an authenticated
`source.json` issued from the campaign's frozen task lineage.

Copy `student/student.example.env` to `student/student.local.env` and configure
the deployment paths. Keep the source signing key and machine configuration
private. `student/pins.env` pins the export and serving tools. Source
revisions must be full commits; both local and remote trainer checkouts must
be clean. SSH credentials come from the deployment environment. Instance provisioning uses the configured provider profile.

| Script | Runs on | Operation |
|---|---|---|
| `setup.sh` | Training host | Install the toolchain, pinned models, trainer, export tools, and independent cleanup service |
| `pilot.sh SPLITS` | Training host | Create unique jobs for each student from `SPLITS/<student>`, train, resume typed pauses, and export |
| `sft-remote.sh SPLITS` | Training host | Train and export each student on the configured machine and return verified training artifacts, checkpoints, and the three exports |

Use `student/machine.example.json` for an existing SSH-accessible machine, or
`student/provisioned-machine.example.json` (Scaleway) or
`student/vastai-machine.example.json` (Vast.ai) to rent one. All use the same
remote job protocol. Configure the SSH user, port, key path, workspace, `uv`
executable, and device. Set `llama_commit` to `LLAMA_COMMIT` from
`student/pins.env`. Jobs use separate workspace directories.

A job measures its machine against the tier's entry in
`config/student-requirements.json` before any work starts. An existing machine
must already have Git, Python, rsync, uv, CMake, a C++ compiler, curl and the
accelerator driver; a rented one is reached as root and has them installed. A
rented machine declares `rental: on-demand`, a `lifetime_hours` bound, and the
GPU memory, system memory and disk it offers; check a configuration before
using it:

```bash
python -m bialy.student_host preflight --configuration MACHINE.json --student 26b-a4b
```

A remote job exports on the machine that trained it, before the machine is
released. The host merge loads the base model in float32, so its peak memory is
about four bytes per parameter: about 96 GiB for 26B-A4B, which fits a rented
GPU machine's memory but not a workstation's. The one merge yields three
exports: the Q4_K_M GGUF for Vulkan in `gguf/`, the MLX checkpoint in `mlx/`
written by the host's converter, and the LoRA adapter in `lora_adapter/`.
`export_receipt.json` binds each by size, tensor count, quantization and
SHA-256, and records that both model exports carry the training chat template.
Those three, the receipt and the merge manifest come back; the merged
checkpoint and the F16 GGUF are deleted with the machine, and the artifact
records `unquantized_retained: false`. The campaign's export stage accepts that
artifact after rechecking the receipts against the returned files and the pinned
llama.cpp revision.
Existing machines are never deleted, stopped, or registered for instance cleanup.
The campaign defaults pilot training to the full training destination; an explicit
`pilot_training` setting can select another destination.

Each job preserves its input inventory, attempt logs, terminal status, and
adapter inventory. Retries require a new typed terminal receipt; OOM and
unknown process outcomes do not trigger another launch. Checkpoints are copied
while training and verified before publication. An incomplete transfer cannot
replace an intact generation.

The independent reaper must be healthy before allocation. Leases identify exact
allocated servers, volumes, and addresses. Cleanup remains unconfirmed when the
provider cannot establish deletion, including an allocation whose response was
lost. The reaper keeps reconciling those records, and each rented machine ends
its own rental when the controller stops extending its deadline;
`docs/running.md` ("Rented training machines") has the timings and what each
provider still bills. Returned adapters are accepted
only after dataset, model, context, and terminal receipts match the job plan.
Export separately checks merged weights, GGUF checksums, and a successful load
receipt; that receipt does not establish behavioral qualification.

`tests/test_identity.py` refuses home directories, logins on real hosts, and
private addresses outside the factory network in tracked files. Deployment
identity exclusions can be added to ignored `config/identity.local.txt`.
