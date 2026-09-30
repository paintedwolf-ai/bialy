# Checking a release without a GPU fleet

A release says what it contains, how it was made, and how good the heads
trained on it are. None of that needs to be taken on trust: each claim can be
checked from the release itself, most of it on a laptop.

```bash
git clone https://github.com/paintedwolf-ai/bialy && cd bialy
uv sync --extra hub
uv run bialy fetch dataset --version v1 --out dataset-v1
```

## 1. The files are what the release says (seconds, offline)

```bash
uv run bialy verify dataset --release dataset-v1
```

Every file matches `SHA256SUMS`, the release is complete, and the card
carries its Hub metadata. `SHA256SUMS` itself matches the release's anchor:
`releases/dataset-v1.json` in this repository, committed when the release was
built. Anything inside a release can be rewritten and checksummed again by
whoever holds a copy; the anchor cannot, so a mirrored or re-uploaded copy that
was changed fails here. Clone this repository from its canonical home, and
check a signed tag if you want the anchor's own history vouched for.

## 2. Every number is recounted from the rows (seconds, offline)

```bash
uv run bialy audit dataset --release dataset-v1
```

`audit` recomputes, from the rows alone:

- **Shape.** Every row matches `row.schema.json`, the schema the Painted Wolf
  Code commit named in `PROVENANCE.json` exported the rows with.
- **Statistics.** Rows, hosts, languages, partial turns, prompt groups, and
  repositories per split, compared with `PROVENANCE.json` and the card.
- **No leakage.** No prompt group (a request and its near-duplicates) appears
  in two splits, and every held-out repository appears only in the test
  split, so validation and test numbers measure requests and codebases the
  heads never saw. The held-out repositories come from the anchor, and
  `PROVENANCE.json` must name the same ones.
- **Tasks.** Every row comes from a task `tasks.jsonl` lists, and the task
  outcomes match `PROVENANCE.json`; `tasks.jsonl` is what a rebuild drives.
- **Judge agreement.** The quadratic-weighted kappa between the two judge
  families, recomputed from `agreement.jsonl`, the second-judge sample the
  release ships, and compared with what the card states.

It exits non-zero and names the first claim it cannot reproduce.

## 3. The judge's scores reproduce (one GPU, or a CPU with patience)

Serve one of the judges `PROVENANCE.json` pins, at its pinned revision, on
any OpenAI-compatible server (vLLM, SGLang, llama.cpp's server, and others),
then re-score a random sample of the rows that judge scored:

```bash
uv run bialy audit dataset --release dataset-v1 \
  --rejudge 50 --endpoint http://127.0.0.1:8000/v1 --model gemma-4-26b-a4b-it
```

The re-judge uses the release's own `corpus.json` for the card texts, the
judge's exact prompt, the same seeded candidate order, and greedy decoding,
and reports how often each released score is reproduced (exactly, and as a
weighted kappa). Scores from a different server or quantization can differ
slightly; a large gap is worth a *Data problem* report.

## 4. The heads' results reproduce (CPU)

The heads release ships the replay reports behind its model card in `eval/`.
Replay the heads over the dataset's test split through the Painted Wolf Code
engine and compare:

```bash
uv run bialy fetch heads --version v1 --out heads-v1
# In a Painted Wolf Code checkout at the heads' engine_commit (in their PROVENANCE.json):
#   build bialy (scripts/build-decide.sh) and write a launcher that runs
#   bialy serve --model <laya-multilingual checkpoint> --model-id convaiinnovations/laya-multilingual \
#     --device cpu --head-max-len 512 --head turn-load=heads-v1/turn-load.safetensors --head unit-rank=heads-v1/unit-rank.safetensors \
#     --head code-rank=heads-v1/code-rank.safetensors
uv run bialy audit heads --release heads-v1 --dataset dataset-v1 --lycaon <checkout> --engine <launcher>
```

Every headline metric (tool precision, recall, F1, guide omission precision,
kind accuracy) must match the shipped report within 0.01; the engine is
deterministic, and the tolerance absorbs floating-point differences between
CPU and the GPU or Apple silicon the report was made on.

## What this cannot check

Sessions are sampled, so no one can regenerate the exact rows. What a rebuild
reproduces is the process: the same pinned models, repositories, tasks, and
engine build, with this repository at the tag the release names, as its
`PROVENANCE.json` recipe records them.
Steps 1–4 establish that the released rows are internally consistent, that
the judge scores reproduce, and that the published results follow from the
released heads and rows.
