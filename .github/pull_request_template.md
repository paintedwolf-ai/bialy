<!--
  Pull requests are currently restricted to collaborators, and code changes need an
  agreed issue first. See CONTRIBUTING.md.
-->

## Which is this?

- [ ] Documentation, wording, or a failing test for an already-filed bug — no issue needed
- [ ] Code change linked to an issue labelled `accepting-work` — Fixes #

## Summary

<!-- What does this change do, and why? Does it change what a release contains? -->

## Checklist

- [ ] Every commit includes a DCO `Signed-off-by:` trailer (`git commit -s`)
- [ ] `uv run ruff check src tests`, `uv run pytest -q tests`, and `uv run bialy check` pass
- [ ] If it changes what a release contains, the effect is described above
