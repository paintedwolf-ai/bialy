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
