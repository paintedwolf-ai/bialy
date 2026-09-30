# Training host

Scripts that train and evaluate the decision heads on a rented GPU host. They
run the Painted Wolf Code trainer (`scripts/decide/` at a recorded commit)
over rows this repository produces; everything they read is an input you can
name and hash.

| Script | Does |
|---|---|
| `setup.sh LYCAON_TAR DATA_DIR` | Installs the trainer and the pinned backbone under `$R` (default `/scratch/bialy`) |
| `jobctl.sh start NAME -- CMD` | Runs every job in its own process group; `stop` ends the whole group, `status` and `wait` report; a name is used once |
| `train.sh GPU RECIPE RUN` | Trains one head into `out/RECIPE-RUN` from a named recipe |
| `eval.sh GPU NAME TURN RANK KIND CORPUS ROWS TRUTH [ROSTER]` | Calibrates, replays, measures discovery, or saves `rank-dump` scores through the CUDA engine |
| `mix.py` | Adds generated skill-request families to the session rows for training |
| `corpus_with.py` | Writes a corpus copy with decision thresholds replaced, for replaying a calibration |

The trainers lock their output head for the life of the run, write it
atomically, and train at the backbone checkpoint's full context; a head
records the encoding it trained with, and `scripts/decide/parity_probe.py`
refuses a head whose encoding differs from what the engine serves.

## Release workflow

The shipping release is B5 turn-load, E4-dense1 unit-rank, and open1 code-rank.
B5 uses independently encoded tool options and a 0.94 initial-preload cutoff.
The 0.94 policy admits 161 tools over the 445 validation turns (0.95 admits 137);
consensus precision/recall are 79.8%/31.9%. These are validation selection results.
E4 automatically preloads at most one skill scoring at least 3.4. Free-text skill
lookup uses 3.2, remains separate, and the full skill catalog is not injected into context.
Turn optimization runs for every workflow and worker assignment. Workflow
manifests do not configure it; unavailable engines preserve applicable guidance.
B2 remains the archived reference baseline.
Tool preloading stays enabled. Request retrieval loads the selected tools without
family or companion expansion. Guide omission and the answer-only veto stay off.
Skill discovery is evaluated against free-text needs over the full loaded catalog;
initial preload precision does not establish discovery quality.
Run `skill_probe.py` with `skill-probe-cases.json` against both the full corpus and
a project roster (`--roster` is a JSON list of names). Keep the raw rankings and
head hashes. The cases are regression diagnostics, never training rows or a fresh
acceptance set; adding packs can introduce highly scored distractors.

The release goal is useful behavior in Painted Wolf Code: modest initial tool
selection, reliable retrieval when the coordinator asks, and no silent mismatch
between the trained head and the host. Initial tool prediction need not be perfect.
Compare with the installed baseline and previous heads on the same requests, and
report both useful coverage and unnecessary schemas. Do not turn a balanced set of
rare-tool examples into a claim about normal request frequency. Clear Git,
container, file, browser, image and other cases are diagnostic coverage; they do
not get fed back into training or replace the original validation distribution.

For the existing baseline, restore the immutable training inputs from the
`20260929-recovery/final` archive.
`data/final-inputs/SHA256.json` identifies E4's prepared data; trainer
`946aea8506c9440a662f87a8d5479963f27735d4` and factory recipe `b60c151`
produced E4. The archive includes B2 and open1-code artifacts and their earlier
provenance. Preserve those weights when retraining only unit-rank. Exact release
restoration uses the archived weights; retraining reproduces the experiment,
without promising byte-identical accelerator results or regenerated judgments.

The independent-encoding controls use the original session splits archived in
`20260929-tool-loading`, trainer `3d79a91c92`, and factory recipe `bf0ee33`.
They change negative sampling coverage and learning rate, not positive frequency.
After `setup.sh` has prepared `$R`, submit training and dependent evaluation with
unique names:

```bash
./jobctl.sh start B5-release -- ./train.sh 0 B5 release
./jobctl.sh start B6-release -- ./train.sh 1 B6 release
./jobctl.sh start validate-B5-release -- ./turn_validate.sh 0 B5 release
./jobctl.sh start validate-B6-release -- ./turn_validate.sh 1 B6 release
./jobctl.sh wait validate-B5-release validate-B6-release
```

`turn_validate.sh` waits for its training job to succeed, then saves trainer-forward
probabilities for **every complete original validation row** and scores both judge
consensus and observed calls. It does not calibrate on acceptance. Native runtime
inference, a prediction-level parity comparison, and fresh tasks still precede
promotion. Preserve rejected candidates and keep the installed baseline when a
candidate fixes encoding but loses practical usefulness.

A final release archive contains all three selected heads, a head manifest, the
host decision catalog, input hashes, source commits, resolved dependencies, raw
predictions, the selection decision, verification receipts and a rollback copy.
Painted Wolf Code's `lycaon/config/packs/painted-wolf/platform/host/decision-release.json` pins the complete selected set and its initial-preload vocabulary.
`release_manifest.py --corpus TRAINING_CORPUS --heads HEAD_DIRECTORY --release NAME
--out RELEASE_JSON` creates it from the actual weights and evaluated corpus.
The host uses those option texts only for tools the surface permits; new tools
remain fully requestable without disturbing existing preload scores. Refreshing
the vocabulary is a new release input requiring evaluation. The independent-corpus
preparation drops any prior bound vocabulary and builds options from current cards.
Set `BIALY_HEADS_DIR` to the restored artifact directory when building; all
three required heads must match their hashes, labels and backbone. For an isolated
candidate build, update the release manifest in that checkout with the candidate's
full set and encoding settings; the build and host consume the same file. Neither a partial cache nor an empty manifest may substitute the base head.
Commit authored code and the release record locally; publishing is a separate
explicit action. Copy and checksum-verify every needed GPU artifact on the NAS
before powering the host off.

## Guide omission: B7 and the relabelled rows

The turn head answers the guide question again. `train.sh GPU B7 RUN` trains
the B5 recipe with `--families tools,guides`: every tool and every guide
option on its own row, so the head's answer for one unit never depends on the
roster. `B7G` trains the guide rows alone, the fallback for a second engine
slot when a joint head loses tool precision.

Guide labels are relabelled before training, not re-judged. A unit's label is
whether the turn called one of the tools it attaches to or one it declares
`needed_with` (the host's `GuideLabels`, and `relabel_guides.py` for archived
rows, whose call set is rebuilt from the loadable calls plus the attached
tools of every unit labelled true). This gives `claim-evidence` and
`survey-first-pass` labels for the first time; they were unlabelled in every
earlier release, which is why no earlier audit could certify them.

    python3 relabel_guides.py data/corpus.json dist/dataset-open1-b5-e4/train.jsonl data/judged/train.jsonl

`turn_probe.py` scores every tool and guide option of a head with the
trainer's own forward, tools and guides alike, and writes rows `guide_audit.py`
and `turn_score.py` read, so selection needs no engine build on the host:

    ./jobctl.sh start B7-release -- ./train.sh 0 B7 release
    ./jobctl.sh start B7G-release -- ./train.sh 1 B7G release
    python3 turn_probe.py --trainer $R/lycaon --corpus $R/data/independent-corpus.json \
      --rows $R/data/judged/holdout.jsonl --head $R/out/B7-release/turn-load.safetensors --out $R/eval/B7-release/holdout.predictions.jsonl
    python3 guide_audit.py $R/data/judged/train.jsonl $R/eval/B7-release/holdout.predictions.jsonl $R/eval/B7-release/guide-selection-holdout.json

The audit certifies a unit at 97% omission precision and 98% needed-guide
retention on both validation and the held-out repositories. Labels are
observational: a unit's positives are the turns that called its tools, so a
unit whose tool is rarely called (recall) can omit with high precision and
still fail retention on a handful of positives. Ship decisions that go past
the audit are recorded in `decisions.yaml` beside the numbers, never silently.

## Rebuilding the heads

1. **Rows.** Session rows come from `bialy collect` and `bialy split`.
   Judge every unit with the hosted pair (`config/models.yaml` role `judge`):

       bialy judge --corpus corpus.json --rows split/train.jsonl --out judged/train.jsonl --second-fraction 1.0

   and the same for `val` and `holdout`. Generated skill requests:

       bialy skillreq --corpus corpus.json --out skillreq/train.jsonl --writer glm-5.3-flash \
         --families clear=4,nearmiss=2,multi=2 --none-families 40 --per-family 5 --seed 7 --repo <each training repository>
       bialy judge --corpus corpus.json --rows skillreq/train.jsonl --out skillreq/train.judged.jsonl --unit skills --second-fraction 1.0

   An acceptance set is written the same way by the other writer, on the
   held-out repositories, with `--split accept --offered-from judged/holdout.jsonl`,
   and judged on `--unit skills --unit tools`.

2. **Host.** `setup.sh` with a tar of `scripts/decide` and `scripts/artifact_paths.py`
   and `lycaon/config/packs/painted-wolf` from the Painted Wolf Code commit,
   and a data directory holding `corpus.json`, `original/`, `judged/`, and
   `skillreq/`. Replay reads the committed tool-schema and guide files to
   estimate standing prompt bytes; the trainer-only tar cannot supply those
   sizes. Build the engine from a full checkout at the same commit with
   `BIALY_FEATURES=cuda ./task build:decide` from its root, using the CUDA
   toolkit the driver supports. The trainer-only tar is not a complete engine
   build tree; the native crate embeds sibling repository files.

3. **Train**, one job per recipe:

       ./jobctl.sh start B2-r1 -- ./train.sh 0 B2 r1
       ./jobctl.sh start E1-r1 -- ./train.sh 1 E1 r1

4. **Select** on validation: `eval.sh ... calibrate` per candidate and label
   view, then replay and discovery on the selection rows with each candidate's
   calibrated thresholds (`corpus_with.py`). Evaluate the frozen choice once on
   the acceptance set.

5. **Confirm on the shipped runtime.** Replay the chosen pair through the MLX
   engine on the same rows and compare predictions, not only aggregates.

Copy `out/`, `eval/`, `jobs/`, and `logs/` off the host before powering it
off: power-off erases `/scratch`.

## Recovery experiment, 2026-09-29

Keep the installed E file immutable. E0 trains on revised session labels and
worker inputs recovered from their recorded job bindings;
E1 and E2 add all or half of the generated training families. Generated eval
families never train. Candidate selection scores every head on the same
selection rows; final acceptance scores only the frozen winner and baseline.
All three recipes also use that shared selection set for early stopping.

E3 and E4 repeat E1 and E2 with eight scored pairs and twelve zero-score
pairs per request, instead of four and three. E5 repeats E4 using judged
levels alone, without promoting observed skill reads. These recipes test
whether broader negative coverage reduces confident confusions without
requiring more hosted labels. They use the same selection set; the fresh
acceptance set remains unopened until a candidate and thresholds are frozen.

`rank_eval.py dump --trainer CHECKOUT --corpus FILE --rows FILE --engine LAUNCHER
--head-file turn-load=TURN_HEAD --head-file unit-rank=RANK_HEAD --out FILE` saves
all card scores once. Name every loaded head, including code-rank when present.
Use immutable head copies from completed training jobs: their hashes and the
launcher are bound into resumable prediction manifests.
`rank_eval.py score --rows DUMP --train
SESSION_TRAIN --roster ROSTER --out REPORT --calibrate` selects preload and
request thresholds on selection data. Omit `--calibrate` on acceptance data.
Skill-frequency bands always derive from session training labels, so augmentation
does not move previously rare skills into the common band. No-skill false
preloads require complete agreed negative labels; missing or disputed labels
are not evidence that no skill applies. Keep raw predictions for runtime parity.
Preload reports also distinguish judge disagreement, possible relevance, and
agreement that a skill is unlikely to help. Strict consensus precision remains
visible, but these categories support reviewing useful alternatives and actual
task outcomes instead of treating every non-consensus pick as equally harmful.

The job controller records terminal exit codes. A missing exit record means
interrupted, not success. Its `wait` command fails for failed or interrupted
jobs. Run names and output directories are never reused. Verify controller
survival and exit codes on cheap jobs before renting a training host.

Long judging passes use a saved JSON manifest and `judge_batch.py MANIFEST JOB`.
The manifest records `factory_commit`, `corpus`, and a `jobs` mapping whose entries
name `rows`, `out`, `workers`, and `units`. Paths resolve relative to the manifest.
Supply `FIREWORKS_API_KEY` through the environment; never save it in the manifest.
Run from the recorded factory commit. Each output has a lease, checkpoint,
input/prompt signature, and terminal result receipt. Exit without a successful
receipt is not completion. The earlier recovery jobs used the equivalent direct
`judge.run` wrapper, preserved with their manifests; new runs use this entrypoint.
The wrapper also saves provider token usage beside each output, including reported
reasoning and cached-token counts, without prompts or credentials. Set
`BIALY_USAGE_JSONL` to choose a shared ledger for other factory commands.
Use current provider rates to estimate costs; requests without a returned usage
record can still be billed. Account/authentication refusals stop pending judging
work and leave completed consensus rows resumable.

For a recovery run, use `prepare_rank_data.py JUDGED_SESSIONS JUDGED_AUGMENT OUT`
to build `judged/`, `skillreq/`, and one shared `selection.jsonl`. This excludes
exact session train/validation text duplicates from validation and entire generated
training families whose text overlaps selection. Its report records exclusions.
Keep the raw judged files and this report in the archive.

For older worker rows that contain a rendered assignment preamble, run
`recover_worker_requests.py WORKER_ROWS ARCHIVED_DATABASE_ROOT OUT REPORT`.
It matches opening-message and parent/child job identities, using the direct
job's brief or a delegation job's leg prompt. It refuses conflicting records
and skips databases with uncheckpointed WAL files. Re-judge recovered rows on
skills and tools, then merge by session/receipt identity while preserving the
original file split. Never infer a split from historical row metadata alone.

`turn_dump.py` saves raw turn predictions, and `guide_audit.py` selects an
explicit guide-omission allowlist on validation. Unknown guide labels do not
certify omission. Confirm frozen per-unit precision and needed-guide retention
on held-out rows; if confirmation fails, keep guides included. The recovery
release disables guide omission after that check failed.

Capture a trainer tar from the recorded lycaon commit, then use `setup.sh` with
the prepared data and corpus. `requirements-cuda.txt` pins the main training
packages; setup saves the fully resolved package versions. Preserve those too.

Calibrate with `rank_eval.py score --calibrate` only on selection predictions.
On acceptance use `--thresholds SELECTION_REPORT` to reuse all selected values.
`accept_rank.py BASELINE_REPORT CANDIDATE_REPORT OUT` applies the frozen skill
limits. Recorded request needs provide a separate regression gate when generated
acceptance rows contain no observed tool requests. `rank_parity.py` compares saved
CUDA and MLX scores and decisions without rerunning calibration. Pass its
`--selection REPORT` option to compare request decisions at the chosen thresholds.

Archive the chosen and rejected heads, rows, input manifests, raw predictions,
reports, source bundles, resolved dependencies, and job logs. `manifest.py write
ROOT MANIFEST` and `manifest.py verify ROOT MANIFEST` check regular files, including
extra-file detection; symlinks are excluded. Never include API keys. Regenerating
hosted labels can vary despite seeds, so archive judged rows to reproduce a chosen
training run rather than promising bit-identical provider regeneration.

`tool_event_rows.py SESSION_VAL OUTPUT` creates evaluation-only variants using
each recorded loadable tool call and the host's bounded request-plus-tool text.
The export does not retain call order, so these are sensitivity probes, not
faithful first-event replays or fresh event-specific labels. Review them before
choosing the separate `tool_event.preload_at` threshold; never train on them.

E6 is E3 with mined training negatives. First dump E2's predictions on the full
E1 training rows, then run `hard_skill_rows.py --rows E1_TRAIN --predictions DUMP
--out $R/derived/hard-skill-train.jsonl`. It verifies the prediction manifest's
source hash and all row identities, keeps every nonzero judged pair, and keeps
eight highest-scoring zero pairs plus four seeded random zero pairs per row.
Both judges' original values are retained; full labels stay in the source
archive. No selection or acceptance rows enter this derived training set.
`train.sh GPU E6 RUN` consumes it. Record the mining head and derived-file hashes
from the generated provenance file alongside the usual run manifest.

`rank_forward_probe.py` compares the trainer forward with native rank scores on
a small saved diagnostic sample. Keep its score gaps separate from the full
CUDA/MLX acceptance comparison; a sparse forward probe is not a release pass.

For complete-pair runtime confirmation, dump the frozen turn head with
`turn_dump.py TRAINER CORPUS ROWS LAUNCHER OUT turn-load=PATH unit-rank=PATH
code-rank=PATH` on both machines, listing every loaded head. Then run
`turn_parity.py CUDA_DUMP MLX_DUMP REPORT --threshold LOAD_AT`. This covers tool
loads; the release keeps all guides and disables the kind veto. Use
`rank_parity.py` separately for skills and request ranking.

Listing and preloading have separate costs. `rank_eval.py score --list-at VALUE`
measures the display cutoff without lowering the preload bar. Choose both on
selection rows, preserve `list_at` in the frozen selection report, and pass that
report to acceptance and `rank_parity.py --selection`. The latter compares the
actual display cutoff as well as preload and request decisions.

## Frozen recovery choice

The 2026-09-29 choice is B2 turn-load + E4-dense1 unit-rank + the existing
open1 code-rank. E4 uses trainer `946aea8506` and this repository's
`b60c151` recipe: the seeded half-family augmentation mix, eight scored skill
pairs and twelve zero pairs per row, and `skills-blended` levels. The backbone
revision is `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`.

The complete run is preserved in the `20260929-recovery/final` archive.
Use `data/final-inputs/` and its hashes as the training payload, the recorded
trainer snapshot, and `train.sh GPU E4 NEW_RUN`. Preserve the new run separately;
do not overwrite the archived heads. `reports/frozen-choice.json` records all
three selected head hashes and `reports/frozen-policy.json` records the policy.
`data/release-heads/` is the selected set for `BIALY_HEADS_DIR` when building
Painted Wolf Code through its `./task` wrapper.

Selection fixed the initial skill preload at 3.99, listing at 0.5, tool-event
preload at 3.8, and request ranking at 1.95 with four maximum loads and one
nearest fallback. B2 tool preloading stays at 0.89; guides remain included and
the kind veto stays off. No acceptance result changed these thresholds.

On the 789 fresh requests, relevant-skill visibility among the 660 requests
with a consensus-relevant skill rose from 68.5% to 92.1%, and 122/123 preloads
were relevant. The strict common-skill regression check failed: 92.2% became
87.9%, despite large rare/mid gains and no category collapse. The user had
explicitly prioritized practical overall quality over mechanical cutoffs;
`reports/quality-decision.json` documents that exception, while
`reports/acceptance-verdict.json` retains the failed strict result. Future runs
should report the same trade-offs rather than silently weakening a gate.

Runtime parity retains the strict raw-score verdict and also reports visible
top-six/top-eight set changes, preload-identity changes, and far cutoff
crossings hidden below both rosters. Inspect these alongside the raw predictions
and judged outcomes; a hidden-card diagnostic does not silently turn a failed
strict check into a pass.

## Roster-independent tool heads

Joint tool questions share a fixed token budget. At 512 head tokens, increasing
70 candidates to 71 shortened every option and caused large score changes on
unchanged requests. Check this before tuning thresholds: `turn_audit.py` bypasses
the replay cache and compares repeated calls, tools scored alone, and an added
catalog option. It records raw answers, head hashes, input hashes, and a failing
exit status when the score changes exceed the specified tolerance.

```sh
python train-host/turn_audit.py --trainer CHECKOUT --corpus corpus.json \
  --rows judged/val.jsonl --engine LAUNCHER --head-file turn-load=HEAD \
  --out roster-audit.jsonl --limit 24
python train-host/independent_corpus.py corpus.json data/independent-corpus.json
./jobctl.sh start B3-r1 -- ./train.sh 0 B3 r1
./jobctl.sh start B4-r1 -- ./train.sh 1 B4 r1
```

B3 uses consensus labels; B4 uses observed calls, matching the earlier A2 label
view. Both use one tool per encoded row, all known positives, and eight seeded
negative samples per training turn. Inverse sampling weights account for the
omitted negatives. Validation retains all known options. The trainer records
`tool_encoding`, the option word bound, and the sampling setting in the head.
It needs the independent-option trainer and matching native protocol; an older
joint head must not be served with the independent flag. These heads train only
tool relevance, so guidance omission and the kind veto remain disabled.

Select on validation using both label views, preload count, and useful-tool
coverage. Check diverse clear requests and no-tool requests, and compare the
same questions with extra or removed candidates. The user's live requests are
smoke checks, never training inputs. Confirm trainer/native and CUDA/MLX parity
before installing. Preserve failed audits as well as passing ones.

Request replay needs a fresh host corpus with `request_loads` on each tool.
That export records the schemas activated by the handler: one selected schema
per tool in the no-grouping host. Older grouped exports include families and
companions. Reports without that metadata are explicitly unverified for expansion. Standing-set reports include the floor separately,
assume subsequent turns are warm, and exclude controls added by live resources.
YAML schema file sizes are a storage-size proxy, not provider prompt bytes.

For request retrieval, use `rank_eval.py dump --requests-only` with each pinned
unit-rank head and the same validation rows. This mode records exact-name matches,
ranked scores, and tools called after the request, including requests with no
remaining judged-positive tool. `request_metrics` measures the ranking portion;
`request_surface_metrics` includes exact names. These are different denominators.
The `after` labels describe subsequent observed calls, not a complete human gold
standard for the need.

To isolate host grouping from model selection, run
`request_expansion_eval.py --predictions DUMP --rows ROWS --catalog NATIVE_TOOLS_YAML
--out REPORT`. It compares no expansion, resource-family expansion, and families
plus companions on identical predictions. Pin the original catalog to reproduce
the old behavior: the no-grouping host removes `request_companions`. This replay
counts schema choices, not end-to-end task success or resource-implied controls.
Do not infer that a rare tool needs more weight solely because it has few training
positives. Keep any augmentation modest and evaluate on the original session
mix, including no-tool requests.

### Host changes after the frozen recovery choice

The user subsequently restored both initial and tool-event skill preload
thresholds to 3.4. Listing remains 0.5 and request ranking remains 1.95 / 4 / 1.
This is a host policy change, not a newly calibrated or accepted model. Preserve
the original frozen reports; do not rewrite their thresholds or claim their
acceptance numbers cover the later policy. The local host also stopped expanding
requested tool families and companions, while retaining resource-implied controls.
Expanded decision cards distinguish newly offered tools from tools retained at a
warm prompt-cache boundary and omit stale scores for retained tools.

For the independent B3/B4 heads, compare native probabilities with the trainer:

```sh
LYCAON_DECIDE_DEVICE=cpu python train-host/turn_forward_probe.py --trainer CHECKOUT \
  --corpus CORPUS --rows VALIDATION_ROWS --head HEAD --limit 6 --out TRAINER_DUMP
python train-host/turn_score.py --trainer CHECKOUT --corpus CORPUS \
  --rows VALIDATION_ROWS --predictions NATIVE_DUMP --out REPORT
```

The scorer requires complete non-partial-row coverage, scores consensus and
observed-use labels separately, and reports out-of-catalog labels explicitly.
A successful numerical or roster audit establishes correct inference, not useful
predictions. Keep diverse manual probes separate from final acceptance and never
promote a head solely because it fixes the encoding problem.

The follow-up independent-tool experiment uses trainer commit `3d79a91c92`
and Painted Wolf Code native/host commit `03e640265e`. Commit `1f29429709` also includes activation receipts and
replay changes. The source archive and incremental Git bundle are in the
`20260929-tool-loading` archive.
The full source archive is standalone; the incremental bundle requires base
`b12895193c`. None of these experimental native changes is installed in the
main development host.

On 445 complete validation turns, at a 0.89 cutoff, B2 had consensus precision
0.757 and recall 0.295; B3 had 0.696 and 0.274. Both B3 and B4 had zero existing-score
changes when an unrelated tool was added on six diagnostic turns. Their trainer
versus Mac probes covered 420 probabilities each, with maximum gaps below 0.002.
These checks establish the encoding fix but do not establish better preload
quality. B2 was retained after these two controls; the later B5 release replaces it.
No new final-acceptance set was spent on these rejected candidates.

### Existing-distribution controls

B5 repeats B3 on identical session rows and labels, with 24 random negative
options per turn instead of eight. Inverse sampling weights remain in place;
there is no per-tool rare-class weighting and no generated data. B6 repeats B5
with learning rate 1e-4 instead of 5e-4. The purpose is to test negative coverage
and optimization stability without promoting rare tools or changing their
empirical positive frequencies. Compare both against saved B2/B3 predictions on
the original validation distribution. Diverse clear examples are diagnostic only,
not a balanced acceptance distribution or training source.

### Final release selection

On the 445 complete original validation rows, B5 at 0.95 preserves B2's
consensus recall (0.291 versus 0.295), improves precision (0.822 versus 0.757),
and loads slightly fewer schemas (0.308 versus 0.339 per turn). Observed-call
precision is 0.547 versus 0.517, with recall 0.143 versus 0.149. B6 did not
justify replacement at comparable preload volume. This selection uses the
existing validation grid; it is not fresh end-to-end acceptance.

The selected heads, rejected controls, original inputs, raw predictions, source
and runtime evidence are in the `20260929-release` archive.
The initial independent-tool head still misses useful preloads on many clear
requests; request retrieval remains the main recovery path. No new generated
examples or hosted judgments were used in this final control experiment.
