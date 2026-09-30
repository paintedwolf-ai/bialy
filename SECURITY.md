# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report privately through GitHub's **Report a vulnerability** flow (repository
**Security** tab → **Report a vulnerability**), which opens a private
advisory visible only to maintainers. Include the affected component, the
commit or release, a description and, where possible, a minimal
reproduction, and the impact you believe it has.

We will acknowledge the report. If we confirm it, we will work toward a fix
and coordinate disclosure when one is available. There is no guaranteed
timeline.

## Supported versions

Fixes land on the default branch (`main`); a published dataset or heads
release with a problem is superseded by a new version, never edited in place.

## Scope

In scope:

- **Runner isolation** — a generated session reaching the host, the model
  servers' host, private network ranges, or the cloud metadata service past
  the runner network's egress rules, or reading anything the runner is not
  given.
- **Credentials** — a token or key reaching a runner, a log, a row, or a
  release.
- **Release integrity** — a way to publish a release that passes
  `bialy verify` or `bialy audit` while its files differ from what its
  checksums, card, or provenance state.
- **Data that must not be published** — a release row carrying personal data,
  secrets, or source from a repository whose license does not allow it.

Out of scope: the behaviour of the open-weights models the factory runs
(report those upstream), and Painted Wolf Code itself (report in its
repository).
