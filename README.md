# Bialy

The open dataset factory for Bialy, the tuned local decision
engine in Painted Wolf Code. It
writes realistic developer requests for pinned public repositories, drives
them through sandboxed Painted Wolf Code sidecars with open-weights models,
exports every turn decision as a training row, has an open-weights judge
score skill and tool cards, and splits the result into train, validation, and
held-out repositories. Factory-authored material is Apache-2.0; repository
excerpts retain their upstream licenses. No proprietary model touches a label
and no private session contributes a row.

Training, calibration, and replay live beside the engine in the Painted Wolf
Code repository (`scripts/bialy/`), because a head must be encoded exactly
as the engine encodes it. This repository hands them rows in their format,
`pw-decide-row/1`. Existing row, head, and release-anchor format identifiers
remain unchanged so archived data and checksum-pinned weights stay readable.

## Built on Laya

Bialy builds on [Laya](https://github.com/NandhaKishorM/laya),
Convai Innovations' open decision model, using its multilingual checkpoint
and training implementation. We credit the Laya authors for that foundation.
Our tuned heads run over the frozen upstream backbone; releases preserve its
model identity, license, and attribution alongside credit for mmBERT-base.
The native runtime in Painted Wolf Code derives from
[laya-candle](https://github.com/Trystan-SA/laya-candle).

## How a dataset is made

1. **Pins.** `config/models.yaml` names the open-weights models, each pinned
   by revision and recorded as reviewed for training use; `config/repos.yaml`
   names the repositories, each pinned by commit, permissively licensed, and
   marked `train` or `holdout`.
2. **Serve.** `bialy serve start` runs one vLLM server per model on the
   GPU host, bound to a private Docker bridge. A model can have replicas on
   another host, reached through an SSH tunnel whose key can forward to that
   one server and nothing else.
3. **Tasks.** `bialy tasks` has the generator models read facts about each
   repository and write requests for each archetype (`config/archetypes.yaml`),
   in several languages, some starting a workflow that plans worker legs.
   Requests naming files the repository lacks are dropped; near-duplicates
   share a prompt group. Workflow tasks carry both `workflow` and
   `workflow_version`, taken from the exact manifest under `runner/workflows/`
   that the runner image installs. This pins replay to the same workflow
   definition; rebasing tasks changes their driving models and preserves that
   identity. Rebuild task files from their raw batches when adopting a new
   workflow version.
4. **Runners.** `bialy fleet run` drives the tasks through runner
   containers: a fresh checkout at the pinned commit with its dependencies
   installed, one sidecar with approval prompts off, and
   `lycaon-debug decide generate` driving each task in its own session. A
   runner can reach the model servers and the public web on ports 80 and 443,
   nothing else: private ranges and the cloud metadata service are dropped,
   and it holds no credential. Each runner exports its sessions' rows with
   `lycaon-debug decide export` and keeps its store for later re-export.
5. **Judge.** `bialy judge` scores every skill card against each turn and
   every offered tool card against each `request_tools` need, 0..4, with a
   model of another family than the one that drove the session, greedy
   decoding constrained to the reply schema, and a seeded candidate order per
   row. A sample is judged by both families to report agreement.
6. **Code-rank pairs.** `bialy coderank` harvests code units from each
   repository with the engine's own parsers, pairs each with requests (the
   unit's leading comment, and requests the generator models write), and
   builds the candidate sets the code-rank trainer reads.
7. **Split and release.** `bialy split` holds out whole repositories and
   splits the rest by prompt group; `bialy release` writes the splits, the
   code-rank pairs, a dataset card, provenance, and checksums. Nothing uploads.

A release lists every task its sessions were driven with (`tasks.jsonl`,
with each task's outcome), so a pass stopped early, restarted, or salvaged
still names exactly what a rebuild drives; the reason for stopping goes in
its provenance (`--stopping`).

A dataset is built in up to two passes, and its rows record which. The first
runs with the decision engine off, so every turn's tool labels are independent
of any head and every loadable tool a turn uses arrives through a
`request_tools` need. The second runs with the first pass's heads loaded
(`--pilot`): its needs are the tools a live head missed, which is exactly what
the rank head ranks in production.

## Running it

### One command

`bialy run` drives a whole pass on one machine and can be left alone: it
builds what it runs on, resumes where it stopped, uploads nothing unless asked,
and ends with a report. It needs Docker, a provider key for the hosted models
in `config/models.yaml`, `uv`, and a Painted Wolf Code checkout at the commit
to build; Go and Rust toolchains are used when present and otherwise run in
pinned images. Settings live in `factory.yaml`'s `run` section; the flags
override them for one run.

```bash
export FIREWORKS_API_KEY=…
bialy run --run pass3 --dry-run            # the stages this invocation would run
bialy run --run pass3                      # build → tasks → drive → judge → pilot heads → drive again → release → heads → report
bialy run --run pass3 --until tasks --task-cap 2 --repo cobra          # a small check of the generation stages
bialy run --run pass3 --task-cap 1 --repo cobra --runners 1 --epochs 1   # the whole chain, small
bialy run-status --run pass3
bialy run --run pass3 --from judge         # rerun from a stage after a fix, discarding later results
bialy run --run pass3 --redo image --rebuild-image   # rerun one stage, keeping the rest
bialy run --run pass3 --push               # the same pass, publishing at the end
```

Stages, in order: `check`, `repos`, `build`, `image`, `corpus`, `tasks`,
`skillreq`, `warm`, `plan`, `drive`, `collect`, `judge`, `judge_skillreq`,
`split_pilot`, `train_pilot`, `engine`, `pilot`, `plan_on`, `drive_on`,
`collect_on`, `judge_on`, `split`, `coderank`, `release`, `train`, `evaluate`,
`release_heads`, `report`. Each writes under `<root>/runs/<name>/` and its
status to `run.json` there; a failed stage stops the run with its error in
`REPORT.md`, and the next invocation starts from it.

- `build` cross-compiles `lycaon`, `lycaon-debug`, and `decide-rerank` for the
  runners and assembles the engine payload (schemas, the pinned git, the
  headless browser, Opengrep) from the checkout; `engine` builds the decision
  engine for this host with cargo. The checkout pins Opengrep releases per
  platform; until it pins a Linux one, `run.scanner: candidate` with
  `run.scanner_candidate` pointing at a linux/amd64 artifact directory from the
  downstream Opengrep repository's `engine/build.py` carries that build, and
  `run.scanner: none` runs without a scanner, which the sidecar then reports
  unavailable.
- The first pass drives with the engine off. `train_pilot` trains the recipes
  on its rows, `pilot` packs them with a linux engine and the checkpoint, and
  the `_on` stages drive the same tasks with those heads deciding. Turn
  `run.engine_on` off (or pass `--no-engine-on`) for a single pass.
- `skillreq` writes requests for every skill and `coderank` harvests units and
  writes request pairs, both through the writer model, so the unit-rank and
  code-rank recipes have their data; `train` runs every recipe in
  `run.train.recipes` under `run.train.max_hours` on this machine's
  accelerator (ROCm, CUDA, Apple silicon, or CPU, with the torch index chosen
  from what it finds).
- `evaluate` replays validation and holdout rows through the host engine
  loading the trained heads, and `release_heads` puts the results on the card.
- With `--push`, `report` commits the release anchors on `release/<version>`,
  pushes it, opens a pull request with `gh`, and uploads both releases to the
  Hub. Without it, the report lists the commands.
- `run.spend_ceiling_usd` stops the run between stages once priced hosted
  models (`hosted.price_per_million`) have used it. Calls the runners' own
  sidecars make while driving are not metered here; the provider's dashboard is
  the record for that part.

The fleet stages mount cache overlays and set bridge firewall rules, so
`bialy run` runs as root from the `warm` stage on; `check` says so when it is
not. To leave a pass running through logouts, sleep, and reboots, install
`deploy/bialy-run.service` (instructions in the file).

### By hand

On the GPU host (Ubuntu with NVIDIA drivers, Docker, and the CUDA toolkit
that vLLM's kernels compile against):

```bash
# The engine the runners carry: Painted Wolf Code built from a clean commit.
# BUILD.json names the commit; every stage records it with the binaries' digests.
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o bin/lycaon ./cmd/lycaon
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o bin/lycaon-debug ./cmd/lycaon-debug
# engine/: schemas/ (the repository's), gitengine/ (./task gitengine:fetch),
#   browser/ (lycaon browser ensure), and opengrep/ (a Linux Opengrep build)

bialy repos fetch
bialy serve start
bialy fleet image --lycaon-bin bin --engine engine
bialy fleet warm      # package caches, once, before any session; runners read them through an overlay
bialy tasks --out tasks
bialy fleet plan --run pass1 --tasks tasks
bialy fleet run --run pass1
bialy collect --run pass1 --pass engine-off --tasks tasks
bialy judge --corpus corpus.json --rows runs/pass1/rows.jsonl --out pass1.judged.jsonl
bialy split --out split/ pass1.judged.jsonl
bialy coderank harvest --decide-rerank bin/decide-rerank --out coderank/units
bialy coderank pairs --units coderank/units --out coderank/pairs
bialy coderank dumps --decide-rerank bin/decide-rerank --units coderank/units --pairs coderank/pairs --out coderank/dumps
bialy release --split split/ --version v1 --code-ref v1 --out dist/v1 --schema <lycaon>/scripts/bialy/row.schema.json \
  --corpus corpus.json --agreement pass1.judged.jsonl.agreement.jsonl --driven runs/pass1/tasks-driven.jsonl \
  --coderank coderank/pairs --stage tasks/provenance.json --stage runs/pass1/provenance.json \
  --stage pass1.judged.jsonl.provenance.json --stage coderank/pairs/provenance.json --stage coderank/dumps/provenance.json
```

`corpus.json` comes from the same build (`lycaon-debug decide corpus`), so
option texts match the sidecars that produced the rows.

The engine-on pass runs the first pass's heads in every runner. A runner's
CPUs answer a turn in seconds where Apple silicon or a GPU answers in a
quarter of one, and past its deadline the engine abstains, so that pass's
image raises the turn decisions' deadlines (the deadline decides whether an
answer arrives, never what it is):

```bash
bialy fleet image --lycaon-bin bin --engine engine --tag bialy-runner-pilot --decide-deadline-ms 60000 \
  --decisions <lycaon>/lycaon/config/packs/painted-wolf/platform/host/decisions.yaml
# pilot/: bialy (built for linux), model/ (the Laya checkpoint), heads/{turn-load,unit-rank,code-rank}.safetensors
bialy fleet plan --run pass2 --tasks tasks-pass2 --pilot pilot
bialy fleet run --run pass2 --pilot pilot --image bialy-runner-pilot
```

A run is resumable: task generation caches every batch, a shard with rows is
never driven again, a restarted fleet adopts its containers that are still
running, and every container the fleet starts is written to the run's ledger
first. `bialy fleet reap --run <name>` removes that run's containers,
found by exact name in its ledger, and nothing else.

Deploy the factory to the GPU host with its commit beside it, so a copy without
`.git` still records which commit it is (`git rev-parse HEAD > SOURCE_COMMIT`
before copying); each stage's provenance also records a digest of the source
that checks the copy matches that commit.

## Reproducing a release

A release names the tag of this repository it was made with, and its
`PROVENANCE.json` carries the recipe read from every stage: each model's
pinned revision and serving arguments, each repository's pinned commit and
license, the Painted Wolf Code commit and binary digests the sessions ran on,
the vLLM version, the configuration each stage read, the seeds, and the
judge's corpus. Where and when a stage ran is not part of it. Sessions are
sampled, so a rebuild reproduces the process rather than the bytes. Released
rows and trainer snapshots preserve the training inputs and recipe; identical
head bytes also depend on the runtime and numerical behavior.

## Checking a release

For the standard release format, use `bialy verify` for integrity,
`bialy audit dataset` to
recount every statistic, prove the splits do not leak, and recompute the
judge agreement from the shipped second-judge sample, `--rejudge` to score a
sample again with a pinned judge on one GPU, and `bialy audit heads` to
replay the heads on a CPU and compare with their model card. See
[docs/verify.md](docs/verify.md).

The historical `open1-b5-e4` bundles have separate per-head configurations
and archived evaluation formats. Their cards describe the supported checks;
the generic dataset and heads audit commands do not establish their validity.
Checksum verification establishes artifact identity, not model quality.

## Development

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest -q tests
uv run bialy check
```

CI runs the same on every push and pull request. It never generates, judges,
or trains: those need GPUs and hours, and run on a GPU host by hand.

## Publishing

Releases go to Hugging Face, named in `config/hub.yaml`: the dataset to a
dataset repository and the heads to a model repository. Each card identifies
the evidence it includes and its acceptance limitations. This repository
holds the factory code, documentation, and checksum anchors in `releases/`.
Large release bundles and private preparation inputs stay in ignored `dist/`.

The prepared `open1-b5-e4` dataset includes the B5/E4 training inputs, original
code-rank candidate inputs, generated tasks, sanitized conversation exports,
historical trainer snapshots, licenses, and source-to-release hashes. Its
overlapping snapshots are preserved separately, not merged into new splits.
The model bundle retains the selected weights unchanged. B5's validation
selection and E4's historical acceptance do not establish current-policy
acceptance; E4's historical common-skill criterion failed.

The Bialy-branded package revision is `open1-b5-e4-bialy`, with separate
dataset and heads anchors in `releases/`. It preserves the `open1-b5-e4`
model identity, weights, and training inputs. The model destination is
`paintedwolfcode/bialy`; the dataset is `paintedwolfcode/bialy-dataset`.

Commit the reviewed checksum anchors before uploading either bundle. Publish
the dataset and model as separate Hub repositories, verify their tagged
contents, then make them public when ready. Preparation does not upload them.
The commands below illustrate the standard release workflow; use the
prepared historical bundle directly instead of rebuilding it with the
generic release commands.

```bash
bialy release-heads --heads heads/ --version v1 --dataset-version v1 --engine-commit <sha> --out dist/heads-v1 \
  --eval val=replay-val.json --eval holdout=replay-holdout.json \
  --baseline val=replay-val-base.json --baseline holdout=replay-holdout-base.json \
  --rerank definitions-zod=rerank-definitions-zod.json
bialy publish dataset --release dist/v1 --version v1          # lists what it would send
bialy publish dataset --release dist/v1 --version v1 --push   # uploads, tags v1
bialy publish heads --release dist/heads-v1 --version v1 --push
```

Publishing needs `uv sync --extra hub` and a Hugging Face token in
`HF_TOKEN`. It checks the release is complete and matches its `SHA256SUMS`,
creates a missing repository as private, uploads the folder as one commit,
and tags it with the version; making a repository public is a separate
choice on the Hub. Repository names come from `config/hub.yaml`, overridden by
`BIALY_HF_DATASET_REPO`, `BIALY_HF_MODEL_REPO`, and `BIALY_CODE_REPO`.

On GitHub, the repository variables `HF_DATASET_REPO` and `HF_MODEL_REPO`
name the Hub repositories, and the `release-publication` environment holds
the `HF_TOKEN` secret. The **Verify release** workflow, run by hand, fetches a
tagged release from the Hub and checks it as a consumer would receive it
(`bialy fetch`, then `bialy verify`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[code of conduct](CODE_OF_CONDUCT.md). If you use the dataset or heads, cite
them with [CITATION.cff](CITATION.cff).

## License

Factory code and authored release material are Apache-2.0. Code excerpts and
test fixtures in candidate inputs or conversation exports retain their seed
repositories' licenses and notices, included in the dataset bundle.
