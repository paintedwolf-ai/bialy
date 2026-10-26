# Running a pass

Why the factory's defaults are what they are, with the measurement behind
each. Pass 1 (dataset v1): 3,024 tasks over 16 repositories, Qwen3.6-35B-A3B
and Gemma 4 26B-A4B on nine vLLM servers. Pass-3 pilots: 53 tasks on two
repositories and two stacks, driven end to end on one AMD desktop through
Fireworks.

## Budgets and shards

Pass 1 gave every prompt 60 minutes, and change work did not fit:

| Archetype | Timed out at 60 min |
|---|---|
| refactor, multi_part | 64–67% |
| bugfix | 52% |
| run_and_verify, small_change | 28–37% |
| question, locate, explain_code, git_history | 0–2% |

A timed-out session keeps every turn it finished, but loses exactly the late
ones (the fix, the verification, the worker legs). So each archetype carries
its own `prompt_timeout`, workflow tasks get `workflow_prompt_timeout`, shards
are cut per budget and sized to `shard_minutes`, and the longest start first.

A shard is stopped well past the sum of its prompts' budgets (each follow-up
gets the full budget). It keeps what it drove: its finished sessions and the
finished turns of the one it cut off; tasks it never reached are listed as
overdue, and the shard is not driven again.

## Preflight and live faults

Runner containers permit the application's own user namespaces, mount isolation,
and syscall supervision. Their security profiles retain the Docker syscall
allowlist and protect system paths through AppArmor. AppArmor must be enabled on
the runner machine. Profiles are content-addressed and loaded before a shard
starts. `SETFCAP` permits the nested namespace to map the container's root user;
the container receives no `SYS_ADMIN` capability. Root and non-root sessions are
both valid execution configurations, subject to behavioral qualification.

The runner places ephemeral application caches under `/tmp/user-cache`. Bundled
self-extracting tools need a writable cache even on their first invocation inside
the application sandbox. Package-manager caches remain under `/cache`.

Container qualification compares representative operations with the same application
build running directly as an ordinary user. Compare effects, typed outcomes,
rejections, and model-visible feedback; paths and generated identifiers may differ.
This comparison must cover the task families admitted to collection. A passing
scanner probe alone does not establish behavioral fidelity. Unattended approval
settings and fixed question answers are explicit collection assumptions, and the
mechanical pilot does not establish equivalence to every human configuration.

Campaign episode collection checks the pinned host's SQLite schema before
starting a runner. Runs selecting `command-v1` require the command program and
child-execution fields consumed by grading and export; a missing field produces
`host_command_receipts_unavailable`. A mechanical pilot may explicitly select
`legacy`, which uses terminal command verdicts and records that surface in its
evidence. It does not qualify revised portable-command behavior. This source
check does not qualify live execution, capture, or the runner image.

Run the source check separately before preparing a pilot:

```sh
python -m bialy.campaign.preflight --host-checkout /path/to/paintedwolf-code \
  --host-commit FULL_COMMIT --out /path/to/new-preflight-directory
```

The directory must be new. It retains the pinned inputs and a terminal result,
including missing receipt fields. A failed check exits nonzero without starting
collection or training. Use a new directory after changing the source.

The mechanical pilot is `python -m bialy.campaign.pilot --configuration FILE`.
Its JSON configuration pins the host checkout and commit, runner image, student
models, llama.cpp revision, serving image, reviewer slots, and storage locations.
The selected `tool_surface` is part of the immutable run input. The runner image
must carry its source commit in the `pw.host_commit` label.

The pilot creates eight tasks per partition: training, validation, development,
and held-out evaluation. It measures two fixed teacher attempts per development
task, retains every outcome, collects whole qualified episodes, exports an
unchanged baseline, trains, exports the
adapter result, and drives paired host evaluations with thinking off. Its result
separates pipeline completion from behavioral success and never qualifies a model
for promotion. The multi-turn probe does not establish long-context behavior.

Session and response reviewers receive explicit JSON schemas whose evidence
references are restricted to the supplied receipts. Returned references are
validated again before admission. Unexpected stage failures retain private
diagnostics beside the run journal and do not trigger blind retries.

Before driving, `preflight` checks the engine-off image and `preflight_on` checks
its engine-on peer. Engine-on probes require the host's decision-engine readiness
fact and a CPU-pinned engine. Both dispatch the pinned scanner over a small fixture.
Custom-model runs also drive a local OpenAI-compatible fixture through the real
adapter, require a committed capture rendered for the declared student, and compare
the image's compaction window with the student's live budget. The host mock marks
its replies scripted, so the capture check uses a separate fixture phase.

Only the host's typed `compaction_summary` request is forwarded to the hosted
teacher, once, using the teacher's original request controls. All other fixture
replies are local. The probe requires a complete summary and an actual context
reduction after the synthesized continuation is published. The fixture uses a
private copy of the image configuration with shorter protected history and a high
automatic-compaction trigger; the student window and provider controls stay intact.
Preflight evidence stays outside the fleet directories consumed by
collection. Custom-model driving refuses missing evidence or a changed image or
settings, including when resuming directly at `drive_on`.

`run.task_timeout: 45m` bounds a whole task, including its follow-ups, and overrides
archetype prompt budgets. `run.tasks_per_runner: 1` gives smoke tasks separate
runners. With no override, the configured archetype budgets and shard sizes apply.

`run.fail_fast.enabled` defaults to true. Every five seconds the fleet reads a
consistent, read-only store snapshot and immutable capture facts. A failed turn,
failed scanner, unavailable required engine, two compaction `length` outcomes,
or no kept payload after three captures stops admission and the run's active
containers. `summary_length_limit` and `capture_turn_limit` set those thresholds.
The abort retains stores, partial exports and a typed `fault.json`; restarting an
aborted pass refuses to spend again. Use a fresh run after repairing the cause.

`bialy run-faults --run NAME` reads active containers or their saved stores and
reports each shard's error-code counts and first occurrence, failed turns, scanner
and engine failures, capture exclusions, modified fields and kept payload counts.
For a collection that should continue after a known model miss, set
`run.fail_fast.task_failure_codes: [turn_closeout_tool_call]` with
`run.tasks_per_runner: 1`. Only that isolated task stops on its first failure.
Its store, partial export, manifest and `task-aborted.json` remain; it counts as
failed and is not retried on resume. Other tasks continue. Unknown submission
codes, provider/driver failures, and any concurrent scanner, engine or capture
health failure retain run-wide scope. The empty default list keeps every
failure run-wide. Failed-task rows are not counted as finished shards. Every receipt candidate
from an aborted task is excluded before quality review; the marker and saved
store remain bound in the input manifest. A resumed pass retains completed and
failed shard counts without re-driving either.

A kept payload is a capture candidate; receipt admission and independent quality
review still decide which candidates train the student. Structured utility health
requires a host build carrying `pw-training-health/1` telemetry under the capture
directory's `health/` subdirectory.

## Workspaces

Every task runs in a workspace: a pinned repository, or a stack
(`config/stacks.yaml`) whose sessions start in an empty directory with git
initialised. Pass 1 had no greenfield rows, and the shipped unit-rank head
scored `write` under the request bar for "create the app files: index.html,
style.css, and game.js". Stack archetypes are written from the toolchain and a
drawn idea and scale (`meta.seed`); repository archetypes that create code may
name new files in directories the repository has. Held-out stacks are held out
like repositories. A stack's `warm` command pre-fills the package cache.

## Workers

Worker rows come from `implement-dispatch` tasks (pass 1: 1 of about 2,860
ordinary implement sessions dispatched a worker), so the archetypes that start
it ask for parts suited to implementer, repo researcher, and web researcher
legs. Legs rarely need a loadable tool (35 judged needed beyond the floor in
340 worker rows); read worker tool metrics with their support.

A leg's decision state is its own assignment, never the user's request: the
leg prompt its coordinator wrote, or a `task()` worker's charter goal. In the
pass-3 pilot the 59 worker states were one-sentence goals (median 225
characters) under the 1,600-character budget. Done-when criteria stay out.

## Models

The driver sets every label a judge does not: guides, kind, the skill a turn
read, and the tools a `request_tools` need went on to use. On coordinator
turns:

| Driver | Loadable calls per turn | Needs whose tools were used | Turns reading a skill |
|---|---|---|---|
| Qwen3.6-35B-A3B | 1.31 | 61% | 11% |
| Gemma 4 26B-A4B | 0.96 | 59% | 2% |
| GLM-5.3-Flash | 2.60 | 90% | 15% |
| Kimi-K3 | 1.78 | 92% | 25% |

GLM-5.3-Flash (MIT) drives and writes alone. In the pass-3 pilot,
DeepSeek-V4.1-Flash reasoned for about 2,700 output tokens a call against
GLM's 225 and its tasks cost five to eight times as much; the same tasks
driven by GLM cost 7.6 times less and ran 18 times faster. GLM is
thinking-only (the provider refuses `reasoning_effort: none`).

DeepSeek-V4.1-Flash judges, with `reasoning_effort: none`. Against low effort
on the same requests it wrote 411 output tokens a verdict instead of 2,222,
matched its own scores on 99% of about 5,800 cards on the likely-or-certain
side, and agreed with GLM as second judge at weighted kappa 0.85 (0.82 at low
effort).

Judges agree poorly on the decision that matters, whether a card is likely or
certain (39–51% for every pair tried, weighted kappa 0.60–0.82); much of it is
real ambiguity between tools that do the same job.

## Fleet

- A runner needs about 2 CPUs and one bridge address; the model endpoint sets
  the pace. The bridge is a /22, and the configuration refuses more runners
  than it holds. Network setup raises the inotify limits every sidecar needs.
- The runners' engine payload must carry git, the headless browser, and
  Opengrep; `fleet image` refuses one without them. Until Opengrep publishes a
  pinned Linux release, runners use a development build (`run.scanner:
  candidate`), and a pass meant for release waits for the pin.
- Engine-on runners decide turns on the CPU, so the engine-on image raises the
  turn deadlines (`run.engine_on.deadline_ms`, 60 s; at 20 s, 14 of 33 rows
  abstained). `collect_on` refuses a pass whose rows all say the engine was
  unavailable.
- A provider refusing the account stops the pass for a resume instead of
  failing every remaining session. Driving spend is not metered here; writing
  and judging are.
- Evaluation replays rows through the host engine one at a time on the CPU,
  about 51 seconds a row, so it replays a seeded sample of each split
  (`run.evaluate.rows`, 500) and runs code-rank sites side by side.
- A snapshot of a live run (`fleet salvage`) can read a shard twice; keep one
  copy per (root session, session, receipt), and merge second-judge samples
  when judging new rows later.

## Labels and training

- Train at the checkpoint's full context. Pass 1's first heads trained at 512
  tokens, where the tool options left about 1% of the request; the engine
  serves at 1,024, and `parity_probe.py` refuses a head whose encoding differs.
- Start from the unweighted tool recipe: rare-tool weighting lowered calibrated
  preload F0.5 (0.37 to 0.28 on called labels, 0.58 to 0.40 on judged).
  Compare candidates by calibrated result and per-tool AUC, never raw
  precision at the default threshold.
- Code-rank trains only on requests that describe a unit without naming it
  (text matching already finds named ones); named requests are kept in their
  own file for measuring the blend. Human-written doc pairs are the neutral
  held-out measure.
- Skills need their own coverage: `skillreq` writes clear, near-miss,
  two-skill, and no-skill requests for every skill, and the rubric scores a
  generic procedure likely only when the request asks for that step. Measure
  discovery with `skill_discovery.py` by band of training support.
- Checkpoints are chosen by tool plus guide validation loss, so selection
  mostly follows guides. More worker rows of the right kinds, not heavier
  ones, is the worker fix.
- Retention rows capture later use as positive observational evidence. Missing use
  is not a negative label. R1 qualification requires verified task truth or an
  independent judge for every held-out label; unverified hindsight cannot qualify
  a head. The host records all candidates when `LYCAON_RETENTION_RECORD=all`.

## Retention pilot

The pilot collects observations and replays experimental candidates. It does
not qualify a compaction head until verified labels and recovery outcomes are
available. The compaction head remains optional and is not part of the runtime
baseline.

```bash
bialy run --run retention-observations --recipes R1 --no-judge --until collect-off \
  --follow-up-rate 1.0 --follow-ups 3 --task-cap 2 --runners 4 --spend-ceiling 40
```

Follow-ups collect longer sessions and canonical recovery observations. Missing
later use is unknown, and successful recall is part of the recovery path. This
collection run does not train a head.

R1 is an explicit experiment, excluded from the default recipes. Training needs
verified positive and negative labels in both splits. Negative labels must have
scenario or judge provenance; repetition and file mutation do not establish
irrelevance. Unjudged observational rows cannot qualify a head.

The search grid trains the optional retention head and evaluates it on separate
validation and held-out workspaces. Replay reports a counterfactual token proxy,
not provider savings, and emits no runtime policy or structural deletion rule.
A deployed comparison must separately demonstrate provider-reported savings,
task continuity, and successful recovery. The host's deterministic compaction
remains complete without the head.

- A run that stops at a failed stage still writes the report; `bialy run`
  with the same name resumes it. When the host or factory code changed under
  a run, `bialy fleet reexport --run <name>/fleet-off --lycaon-debug <binary>`
  rewrites every driven shard's rows from the store it kept, so the pass is
  never driven twice; `train-host/retention_pilot.sh resume` rebuilds, does
  that, and continues the run.
- A prompt that times out is aborted with a two-minute bound; a session that
  will not stop (a dev server the model left running) is recorded on its task
  and the shard moves on, instead of holding the runner until the fleet's
  overdue deadline.
- Rows from stores the earlier pilots kept export the same way:
  `lycaon-debug decide export --db <store.db> --retention rows.jsonl --requests requests.jsonl`
  reconstructs candidates for receipts written before the host recorded
  them (`reconstructed: true`), so label rates can be read before any paid
  call.

## Training hardware

Precomputing the frozen backbone's features dominates. Turn-load on 2,746
rows:

| Device | Precompute | Epoch |
|---|---|---|
| H100 | 47 s | 93 s |
| Radeon RX 7900 XT (ROCm) | 335 s | 178 s |
| Apple silicon (MPS) | about 450 s | about 245 s |

Devices diverge slightly epoch to epoch, so small differences are noise until
repeated. Code-rank does not precompute: about 13 minutes an epoch on an H100.

## Shutting down a GPU host

Power-off erases `/scratch`, where a pass writes everything, so archive and
verify each host's data first, replica-only hosts after the last judge and
training hosts last. Scaleway bills an instance until it is powered off, and
its volumes and IPs until deleted.

### Compaction head readiness

Compaction runs compare the trained head with the host's deterministic context policy. Repeated calls and file mutations do not establish that an earlier observation is expendable. Labels require structured task outcomes, explicit need, or an independent schema-constrained judge; absence of a later citation is not a negative label.

The future head ranks eligible inline detail. Canonical transcript retention, recovery scope, coverage boundaries, and baseline context fitting remain host responsibilities. Missing or untrained heads use the complete deterministic policy. A release must demonstrate useful retention decisions across receiver windows and verify recovery of necessary material; increased successful recall alone is not a failure.

The compaction head is not trained or enabled by these changes. Dataset experiments must record the Painted Wolf Code commit, source scenario, writer and judge families, receiver and summarizer budgets, failure injection, and recovery outcome. Evaluation includes unavailable and stalled summarizers and sessions longer than any receiver window.
