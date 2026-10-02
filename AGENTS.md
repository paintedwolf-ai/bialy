# Bialy — agent policy

**Bialy** is the open dataset factory for the decision heads in Painted Wolf
Code: it writes requests, drives them through sandboxed Painted Wolf Code
sidecars, judges the resulting rows, and releases datasets and heads. Start at
[`README.md`](README.md); why each default is what it is lives in
[`docs/running.md`](docs/running.md), and how releases, tags, Hub revisions,
and Painted Wolf Code versions pin each other in
[`docs/releases.md`](docs/releases.md).

Training, calibration, the row schema, and replay live in Painted Wolf Code
(`scripts/bialy/`). A change to how rows are encoded or read belongs there, not
in a copy here.

## Operating rules

- **Commit only when the user asks.** Stage only your own paths with
  `git add -- <paths>`, and sign off every commit (`git commit -s`); the DCO
  check refuses unsigned commits.
- **Never push or publish without an explicit request in this conversation.**
  That includes branches, tags, pull requests, force pushes, and Hub uploads
  (`bialy publish --push`, `bialy run --push`). `main` accepts changes only
  through a pull request with passing checks.
- **Preserve concurrent work.** Dirty files you did not edit belong to someone
  else: do not revert, reset, or stash them. Never run `git stash` without
  pathspecs. Use a scratch worktree to test `HEAD`.
- **Runs spend money and share hosts.** Never start a run, a judge, or a writer
  against a provider without an explicit request. Keys come from the
  environment and never land in shard files, logs, or commits. On a shared
  host, stop only the processes you started; never kill by name or command line
  (`pkill -f`, `killall`).
- **Report the crux, the change, the verification, and what is unresolved.**
  If a finding contradicts the request, say so before designing around it.

## Testing

For code and configuration changes, run what CI runs:

| Command | Checks |
|---|---|
| `uv run ruff check src tests` | Lint |
| `uv run pytest -q tests` | Tests |
| `uv run bialy check` | The configuration loads and every pin is valid |
| `shellcheck runner/*.sh` | Runner scripts |

Tests use fakes for every model and provider call; no test calls a real model.
Do not write tests that assert documentation wording. Prose-only changes need a
review of the diff and its links, not the test suite.

## Greenfield policy

Internal code can be reshaped directly: rename and update callers, remove dead
code, and update tests and docs in the same change. No shims, alias
forwarders, or backwards-compatibility hedging.

Split large functions along real boundaries, not to satisfy lint. Comments are
short statements about non-obvious logic. Remove stale or obvious comments,
references to other products, historical narratives, and instructions or
lectures. Measurements and the history behind a default belong in
`docs/running.md`, not in comments.

## Durable surfaces

- **Published releases never change.** An anchor in `releases/` and its
  `<kind>-<version>` tag are written once; a different release is a new
  version. `bialy publish` refuses to move a tag.
- **Rows are `pw-decide-row/1`**, defined by Painted Wolf Code's
  `scripts/bialy/row.schema.json`. Change the schema there and bump the row
  version for breaking changes.
- **A model enters `config/models.yaml` only after a person reads its license**
  and records that its outputs may train an Apache-2.0 model (`reviewed`,
  `outputs_trainable`). Repositories are pinned by commit and license text.
- **Painted Wolf Code pins what it ships** in `decision-release.json`
  (`release`, head sha256s, and `source`). Adopting a release updates that file,
  and `docs/releases.md` records which versions ship it.

## Labels and requests

- Request text never names a tool: a head would learn to read tool names out
  of prose.
- Labels come from structured row facts and schema-constrained judge verdicts,
  never from matching prose.
- A row is judged by a model of another family than the one that drove it.
