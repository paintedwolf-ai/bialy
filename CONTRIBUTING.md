# Contributing to bialy

Thanks for contributing. This repository makes the open dataset and the
decision heads for [Painted Wolf Code](https://github.com/paintedwolf-ai); its
code is small, and its outputs are published, so changes are held to the
bar a published dataset needs.

## Pull requests are gated

Right now, **only collaborators can open pull requests**, and code changes
need an issue labelled `accepting-work` first. This is temporary while the
first release stabilises.

## Other ways to help

- **Check a release.** `bialy audit` recomputes everything a release
  claims from its own files, and can re-judge a sample with a pinned judge on
  one GPU (see [docs/verify.md](docs/verify.md)). Report anything it cannot
  reproduce with the *Data problem* form.
- Report rows that are wrong, leak, or carry something that should not be
  published.
- Suggest seed repositories or request archetypes the dataset lacks, in
  Discussions.

## Developer workflow

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest -q tests
uv run bialy check
```

Pull requests must reference an `accepting-work` issue and pass CI. Nothing
in CI generates, judges, or trains; a change to those stages is checked with
the tests' fakes, and a maintainer runs it on a GPU host before a release.

Changes that alter what a release contains (models, repositories,
archetypes, prompts, labels) change the dataset. Describe the effect in the
pull request; they ship with a new dataset version, never inside an old one.

## Developer Certificate of Origin

Contributions are accepted under the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO). It
is a statement that you wrote what you submit, or have the right to submit
it under the project's license; it is not a copyright assignment.

Sign off every commit with `git commit -s`, which adds a
`Signed-off-by: Your Name <you@example.com>` trailer. Use a real name and a
reachable address.
