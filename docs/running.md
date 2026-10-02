# Running a pass

What the first full pass (pass 1, dataset v1) measured, and what the factory's
defaults do about it. Numbers are from 3,024 tasks over 16 repositories,
driven by Qwen3.6-35B-A3B and Gemma 4 26B-A4B on nine vLLM servers.

## Time budgets

A pass is bounded by how long each prompt may run. Pass 1 gave every prompt
60 minutes. With reasoning models decoding about 30 tokens a second per
stream, a single model call took from half a minute to over ten, and change
work did not fit:

| Archetype | Median session | Timed out at 60 min |
|---|---|---|
| refactor | at the cap | 67% |
| multi_part | at the cap | 64% |
| bugfix | at the cap | 52% |
| run_and_verify | 15 min | 37% |
| small_change | 7 min | 28% |
| question, locate, explain_code, git_history | about 2 min | 0–2% |

A timed-out session keeps every turn it finished, so its rows are still
valid, but the turns it never reached are exactly the late ones: the fix
after the diagnosis, the verification after the fix, and the worker legs a
coordinator was waiting on. 64% of the tasks that started a worker workflow
were cut off mid-leg.

Each archetype therefore carries its own `prompt_timeout`, and a task that
starts a workflow gets `workflow_prompt_timeout`. Shards are cut per budget,
so quick tasks never wait behind a long one, and sized by time: a shard holds
as many tasks as fit `shard_minutes` at their budget (eight 60-minute tasks,
two 180-minute ones), so no single shard can run for most of a day. The
longest shards start first: in pass 1 the last 5% of tasks, long change
sessions on the largest repositories, took over an hour on an almost idle
fleet.

## Worker legs

Worker rows come from sessions that dispatch workers, and in pass 1 ordinary
implement sessions almost never did: 1 of about 2,860. Every other worker row
came from tasks that started `implement-dispatch`. Which workers those
sessions dispatched followed from what the request asked for:

| Worker surface | Rows |
|---|---|
| implement (implementer) | 401 |
| explore_readonly (repo researcher, path explorer) | 53 |
| web_research (web researcher) | 1 |

Real use is mostly implement with implementer, repo researcher, and web
researcher legs, so the archetypes that start `implement-dispatch` ask for work
with those three kinds of part: something to look up in upstream
documentation, something to find in the codebase, and something to change.
`research_change` exists for the web-researcher leg specifically.

Worker legs rarely need a loadable tool: the worker floor already carries the
tools a leg works with, and only 35 tools were judged needed beyond it in 340
worker training rows. Tool metrics on worker legs rest on a handful of
positives; read them with their support.

## Fleet sizing

- A runner costs about 2 CPUs of setup and tool work and one bridge address;
  the model servers, not the runners, set the pace. At 224 runners the servers
  ran 20–40 concurrent requests each with 2–35% of their KV cache in use, so
  more runners raise throughput. The bridge is a /22 and the configuration
  refuses more runners than it holds; a /24 stopped at about 250.
- Every sidecar watches its checkout. The kernel's default of 128 inotify
  instances per user ran out a few runners into the fleet, and the sidecars
  after that ran unwatched; network setup raises the limits.
- Throughput settled at 11–20 tasks a minute and fell as the quick tasks ran
  out; the pass took about six hours.

## Greenfield work

Pass 1 drove every session in a mature library at a pinned commit, and its
task writer dropped any request naming a file the repository lacked, so no
row shows a project being started. Real use starts projects often, and the
heads showed the gap: on the shipped unit-rank head, "create the app files:
index.html, style.css, and game.js" scored `write` under the request bar.

Every task now runs in a workspace of one of two kinds. A repository is a
pinned checkout, as before. A stack (`config/stacks.yaml`) is a toolchain, such
as React with Vite, a Python CLI, or a Rust CLI, whose sessions start in an
empty directory with git initialised. Stack archetypes (`new_project`,
`prototype`, `scaffold`, `port`) are written from the toolchain and a drawn
project idea and scale instead of repository facts, so a stack's tasks spread
over many kinds of project; the idea and scale stay on each task as
`meta.seed`. Repository archetypes that create code (`small_change`, `docs`,
`new_module`, `project_setup`) may name files to create, as long as each would
sit in a directory the repository has; the rest still name only files that
exist.

Stacks are held out like repositories: a held-out stack (Vue, Ruby, and Java
with Maven) never contributes a training row, so its rows measure transfer to
a toolchain the heads never saw a project started on. A stack may name a
`warm` command that fills the package cache with what its new projects
commonly install, which saves sessions minutes of downloads without limiting
what they install.

Greenfield sessions run long and reach for tools repository sessions rarely
touch: scaffolding and package installs, documentation lookups, many new files,
running a server or opening the page they built. Their budgets are 90 to 150
minutes a prompt, and `new_project` continues with follow-ups most of the time,
since a new project grows by the next feature. A session cut off by its budget
keeps every turn it finished, and a shard past its deadline keeps its sessions
(see the deadline note under One command on one machine).

## Engine payload

The runners' engine root must carry bundled git, the headless browser, and
the Opengrep scanner. A payload without them still produces rows, from
sessions whose git, browser, and scan tools fail. `fleet image` refuses such
a payload.

## Driving models

The drivers' behaviour still sets every label a judge does not: guides (a
guide is needed when a tool it is attached to was called), kind, the skill a
turn read, and the tools a `request_tools` need went on to use. Measured on
coordinator turns, against the mixed drivers of the earlier heads:

| Driver | Loadable calls per turn | Orientation share of calls | Needs whose tools were used | Turns reading a skill |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 1.31 | 44% | 61% | 11% |
| Gemma 4 26B-A4B | 0.96 | 55% | 59% | 2% |
| GLM-5.3-Flash | 2.60 | 36% | 90% | 15% |
| Kimi-K3 | 1.78 | 28% | 92% | 25% |
| Claude Haiku 4.5 | 0.83 | 35% | 57% | 0% |

Orientation calls are command, command_output, and the git read tools. Choose
drivers on a pilot of a few hundred tasks by these numbers, above all by how
often a driver uses the tools it asks for: a need whose tools go unused is a
positive the rank head never sees confirmed. GLM-5.3-Flash (MIT) is the
strongest candidate and drives alone; it serves from a provider, so runners
need egress to that one endpoint and the key kept out of the sandbox.

Cost per session matters as much, and it follows how long a driver reasons.
In the pass-3 pilot on Fireworks, over about 200 calls each, GLM-5.3-Flash
wrote 225 output tokens a call with 73% of its input cached ($0.0023 a call);
DeepSeek-V4.1-Flash wrote 2,720 with 89% cached ($0.0061 a call), and its
sessions ran longer and made more calls, so its tasks cost five to eight times
GLM's. Its reasoning can be turned off (`reasoning_effort: none`), but driving
without it is unmeasured. DeepSeek judges instead, where the judge already
asks for low reasoning effort. Weigh a second driving family for diversity
against that cost on a pilot before adding one.

## Judging and snapshots

Judges run on a provider, both on every row, and a label counts only what
both agree on (`rows.py` in Painted Wolf Code). Agreement on the decision
that matters, whether a card is likely or certain (3 or more), is low for
every pair tried: 44–47% for Qwen3.6 and Gemma 4, and 39–51% for
GLM-5.3-Flash and Inkling, whose graded agreement is higher (weighted kappa
0.66–0.82 against 0.60). Much of the gap is real ambiguity: tools that do the
same job (edit, replace_lines, write), follow-ups that read without their
conversation, and worker briefs. Telling the tool judge which tools are
already available and to score only the most direct of overlapping tools
halved the tools it called likely or certain without raising agreement.
Next time, also judge guides (a turn's guide labels still follow its calls),
and give worker legs judged tool labels only if a trial shows judges agree
on briefs; on a sample they agreed on none.

A judge's reasoning is mostly cost. The verdict is decoded straight into the
reply schema, and in the pass-3 pilots DeepSeek-V4.1-Flash judging the same
requests with reasoning off instead of low wrote 411 output tokens a verdict
instead of 2,222 ($0.0018 against $0.0040) and finished the stage three times
sooner. Its scores matched its low-effort scores exactly on 90% of about 5,800
cards and on the likely-or-certain side on 99% (weighted kappa 0.80 to 0.83),
and its agreement with GLM-5.3-Flash as second judge rose from 0.82 to 0.85.
It now judges with `reasoning_effort: none`.

A snapshot of a live run reads finished shards and the live stores of running
ones (`fleet salvage`); a shard that finishes between the two reads appears in
both, so keep one copy of each (root session, session, receipt). Judging only
new rows later is fine as long as the second-judge samples are merged too,
since the release's stated agreement is computed from them.

## Code-rank pairs

A generated request that names its unit teaches the head nothing, since text
matching already ranks that unit first, so training reads only requests that
describe a unit without naming it. About half the replies named their unit.
The prompt deliberately does not list the forbidden words, which would steer
the wording away from how people ask; instead the pair writer keeps drawing
units until enough unnamed requests exist, and keeps the named ones in their
own file, a realistic set for measuring the blended ranking.

Training only on unnamed requests is a choice, not a measurement. People often
do type identifiers; whether a head trained on some named requests ranks
better in the blend is an open question the named files can answer, by
evaluating a head trained with and without them.

Doc-comment pairs depend on the repository: some have almost none. On held-out
repositories, model-written evaluation pairs come from the same generators as
the training pairs; human-written doc pairs are the neutral measure.

## Training the heads

Train at the backbone checkpoint's full context. Pass 1's first heads trained
at 512 tokens, where a turn's roughly seventy tool options fill the budget and
about 1% of the request survives, so the tool head learned which tools are
common and not what a request asks; it scored a per-tool AUC near 0.5 on the
common tools and loaded about seven tools a turn. The engine serves at 1,024
tokens, so training and serving saw different inputs. The trainer now always
trains at the checkpoint's context, records the encoding in the head, and
`parity_probe.py` refuses a head whose encoding differs from the engine's.

At the right context, on the same labels and after calibration, rare-tool
weighting (`--tool-weight sqrt-inverse`) lowered tool-preload F0.5 from 0.37 to
0.28 on called labels and from 0.58 to 0.40 on judged labels. The unweighted
recipe performed better in this experiment; start from it and test weighting
again on new data rather than assuming either way.
Tool labels judged by two hosted judges (a tool counts when both score it
likely or certain) give a more restrained head than labels from what the
drivers called, and both reach per-tool AUC 0.80 to 0.90 on held-out
repositories. Judge every candidate by its calibrated result and its per-tool
AUC, never by raw precision at the default threshold.

Skills need their own coverage. Sessions read few skills, the generic ones
(verifying a change, reading history) dominate the judged positives, and the
first judging rubric scored a verification procedure likely for almost any
change. The rank head then listed generic skills ahead of the specialised one a
request needed. The skill rubric now scores a procedure that suits any change
as likely only when the request asks for that step (check its wording against
the disagreements it changes before relying on it: a stricter first attempt
also dropped genuine verification requests and specialised skills), and `pwdecide skillreq`
writes requests for every skill (clear cases, near misses, two-skill and
no-skill requests) that the judges label. Measure discovery with
`skill_discovery.py`, the share of relevant skills a listing or lookup shows,
by band of training support, alongside preload precision.

Checkpoints are chosen by the sum of the tool and guide validation losses, and
the guide loss is several times the tool loss, so selection mostly follows
guides. Weighting worker rows up to balance hosts would multiply labels that
are nearly empty (see Worker legs); the worker fix is more worker rows of the
right kinds, not heavier ones.

### Future improvement: external skill collections

Broaden a future skill-ranking dataset with a substantial, diverse selection of
popular online skills under permissive licenses. This is a future data effort,
not part of the current release's training. Ranking currently sees a skill's name
and short description paired with a request or free-text lookup need; it does not
read the skill's full procedure when scoring.

- Record source URL, repository commit, license identifier and required attribution
  for each imported skill. Keep provenance with the extracted card and retain
  applicable notices; do not assume an entire collection shares one license.
- Remove duplicate and near-duplicate cards. Split by source collection and semantic
  family so repackaged copies cannot cross training, selection and acceptance.
  Hold out entire collections to measure transfer to skills added after training.
- Include realistic requests, paraphrases, multi-step needs, near misses and
  no-skill cases. Judge plausible competing cards without exposing the writer's
  intended answer. Broad coverage must not imply equal real-world frequency:
  preserve session-based prevalence and test any augmentation mix on ordinary use.
- Evaluate both project-sized loaded rosters and expanded catalogs with distractors.
  Report top-result usefulness, relevant alternatives, wrong high-confidence
  matches and per-skill coverage. An extra skill can beat a correct existing match
  even when each card's individual score is unchanged.
- Keep diagnostic cases out of training and final acceptance. In the September 29
  E4 check, a container/build/logs query ranked a Kafka skill above containers,
  and a setup-documentation query ranked Supabase above documentation, across
  61 exported skills. Neither distractor was in the active 53-skill project
  roster; rerunning those exact queries there selected the relevant skills first.
  This motivates expanded-catalog evaluation, not a claim that those live lookups
  failed or that free-text input itself must be reverted.

### One command on one machine

The next pass runs unattended: start `bialy run` on a desktop with a
Fireworks key, come back in a day or two to an anchored dataset, a head set,
and a report. The orchestrator (`src/bialy/run.py`, README "One command")
chains the stages with a resumable state file, builds the runner binaries,
engine payload, and decision engine from the pinned checkout, runs the fleet
as a few containers on the same machine, generates, writes, and judges
through one hosted provider (GLM-5.3-Flash drives and writes, and
DeepSeek-V4.1-Flash, of another family, judges),
drives a second pass with pilot heads deciding, trains all four recipes on the
local accelerator under a time budget, evaluates through the host engine,
stops between stages when a spend ceiling is crossed, and, asked to, commits
the anchors, opens the release pull request, and uploads.

What a small run on an AMD desktop established, and what it did not:

- The chain runs end to end at `--task-cap 1 --workspace cobra --runners 1
  --epochs 1`. Head quality at that size means nothing; the check is that
  every stage's inputs and outputs line up.
- The runner containers hold a shard's deadline, so one hung session cannot
  stall the pass: a container still running well past the sum of its prompts'
  budgets (a request and each follow-up get the full budget) is stopped. The
  shard keeps what it drove, as a timed-out session keeps the turns it
  finished: the sessions it completed, and the finished turns of the one it cut
  off, which the runner's task order identifies. The tasks it never reached
  are listed as overdue, so the shard is not driven again.
- The runners' engine is the CPU build, and turn decisions in a container take
  seconds, so the engine-on image raises the turn deadlines
  (`run.engine_on.deadline_ms`). The deadline decides only whether an answer
  arrives, never what it is. At 20 seconds on 2-CPU runners, 14 of 33
  engine-on rows still abstained on the deadline; the default is now 60.
- An engine-on row records how the engine answered (`engine.state`), and
  `collect_on` refuses a pass whose rows all say the engine was unavailable:
  the first attempt did exactly that, because the pilot's checkpoint lacked the
  host's completion marker, and nothing else in the chain would have noticed.
- Rerunning a stage after a fix is the common case, so `--redo STAGE` reruns one
  stage while keeping the others, a replanned pass discards its stale shards, and
  a judging whose inputs changed clears its resumable manifest instead of
  refusing.
- Driving spend is not metered by the orchestrator: sessions call the
  provider from inside their containers. Generation, writing, and judging are.
- The scanner is the one payload part the checkout cannot yet supply for
  Linux: its downstream Opengrep repository releases macOS builds and pins one
  artifact per platform. The desktop run set `run.scanner: none`, so its
  sessions saw the scan tools report the scanner unavailable; a pass meant for
  release waits for a pinned Linux release, and the fleet provenance records
  which of the two a pass ran with.

### Future improvement: guide labels and the turn state

The September 30 B7 run restored guide omission on labels derived from tool
calls. The next pass regenerates the runs with larger driver models rather
than retraining on the archived rows, so these items are changes to what the
factory records and judges during generation:

- Judge guide relevance directly. A unit's label is whether its turn called an
  attached or `needed_with` tool, so a unit whose tool is rarely called has few
  positives and its retention swings on a handful of turns (`native-recall-tool`:
  16 positives on holdout). A judged 0..4 relevance per unit, like the skill
  cards, would give every unit a label independent of call frequency.
- Give worker legs their structured goal. The host cuts a worker's state to 900
  characters and the task-mode preamble fills it, so the leg's goal rarely
  reaches the head; pass the leg's goal and done-when fields as state.
- Train the tool-event text. The tool-triggered skill read ranks the request
  plus the tool's name and description; no rows record which skill a session
  read after its first loadable tool call. Export that pair from receipts and
  judge it.
- Record `list_dir`, `summarize`, `find`, and other floor calls on rows. Archived
  rows keep only loadable calls, so relabelling reconstructs floor calls from
  sibling unit labels; storing the call list removes the reconstruction.
- Drop the kind family. No host consumer reads it and independent heads never
  answer it; keep the label off rows and the trainer.
- Train a web-rank head before enabling the `web_pages` rerank site; the engine
  refuses to rank with a head it did not load.
- Raise tool-preload recall. B5 holds 0.82 precision at 0.29 recall on
  validation, and on live orientation and edit prompts it preloaded nothing at
  0.85; the model reached every tool through `request_tools` instead. More
  positive tool rows per turn, or judged tool relevance, is the next H100 hour.
- Make request ranking robust to file names and locations. On the shipped
  unit-rank head, "create files" scores write at 3.0, but "create project
  files (HTML/JS/CSS) in the workspace" scores it at 0.5 and "create the app
  files: index.html, style.css, and game.js" at 1.8, under the 1.95 bar.
  Export `request_tools` needs with the tool the turn then used, and add
  judged needs that name files, extensions, and places.
- Measure the prompt, not only the omissions. On a live build session the
  guide head removed 2.2K of 38K system characters; the loadable-tool listing
  is 18K and the floor schemas 22K more. Rows that record which listed tools a
  turn later requested would let a head shorten that listing.

## Training hardware

The heads train over a frozen backbone, so precomputing its features
dominates. For turn-load on pass 1's 2,746 training rows:

| Device | Precompute | Epoch |
|---|---|---|
| H100 | 47 s | 93 s |
| Radeon RX 7900 XT (ROCm) | 335 s | 178 s |
| Apple silicon (MPS) | about 450 s | about 245 s |

The same configuration gives the same starting loss everywhere but diverges
slightly epoch to epoch across devices, so a small difference between two
configurations is noise until repeated. Code-rank trains on every example each
epoch without precomputing: about 13 minutes an epoch on an H100 and most of a
day in total on a laptop.

## Shutting down

Scaleway bills a GPU instance per minute while it is on, standby included; a
full power-off stops that, and its boot volume and public IPs bill until
deleted. Power-off erases `/scratch`, where everything a pass writes lives, so
every host's data goes to the archive and is verified byte for byte before it
powers off. Order the shutdown by what still needs each host: replica-only
hosts after the last judge, then the training hosts.
