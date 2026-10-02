# Releases and the versions they tie together

Three things are versioned separately and pinned to each other: this
repository (the factory that makes datasets and heads), the dataset and heads
releases on the Hugging Face Hub, and Painted Wolf Code, which ships heads and
trains them. Every link is a pin a machine can check, written where the thing
that depends on it lives.

| From | To | Where the pin lives |
|---|---|---|
| Painted Wolf Code | the heads it ships, and the dataset and factory behind them | `lycaon/config/packs/painted-wolf/platform/host/decision-release.json`: `release`, each head's sha256, and `source` (`heads` and `dataset` as Hub repository, tag, and revision; `factory` as this repository, tag, and commit) |
| A Bialy release | the code that built it | git tag `<kind>-<version>` here, on the commit that records the release's anchor; `PROVENANCE.json` and the card name the commit (`factory.git_commit`) |
| A Bialy release | its exact bytes | `releases/<kind>-<version>.json`, the anchor: the sha256 of the release's `SHA256SUMS` |
| A Bialy release | Painted Wolf Code | `engine_commit` (heads) and the recipe's `lycaon_commit` (dataset): the Painted Wolf Code commit the sessions ran on and the heads trained with |
| A Hub release | its version | a Hub tag named after the version, pushed by `bialy publish` |

So from a Painted Wolf Code build, `decision-release.json` names the Hub
revisions it downloads and the Bialy tag that made them; from a Bialy tag, the
release's provenance names the Painted Wolf Code commit it trained against.

## Making a release keeps the chain

1. `bialy run` (or the stage commands) builds the release; its provenance and
   card record this repository's commit.
2. `bialy run --push`, or `bialy publish`, commits the anchor on
   `release/<version>`, tags that commit `<kind>-<version>`, pushes the branch
   and the tags with a pull request, and uploads to the Hub, which tags the
   upload with the version. A tag that already names other code is refused:
   a released version is never re-made.
3. Painted Wolf Code adopts a heads release by updating `decision-release.json`:
   the heads' sha256s from the release, and `source` from the Hub tags and the
   Bialy tag. Its staging refuses a manifest whose source is incomplete, and CI
   downloads the heads from `source.heads`.
4. Add the release, and the Painted Wolf Code versions that ship it, to the
   table below.

## Reproducing a release

Check out the tag the release names, and Painted Wolf Code at its
`engine_commit` (heads) or recipe `lycaon_commit` (dataset). Fetch the dataset
from the Hub at the revision `source.dataset` pins, or rebuild it with
`bialy run` from the tag; sessions are sampled, so a rebuild reproduces the
process rather than the bytes (`docs/verify.md`). Heads train with
`train-host/train.sh` from that dataset, and `bialy verify` checks a release's
bytes against its anchor.

## Ledger

### Heads

| Version | Tag (commit) | Hub tag (revision) | Dataset | Trained with Painted Wolf Code | Shipped in Painted Wolf Code |
|---|---|---|---|---|---|
| open1-b5-b7g-e4 | `heads-open1-b5-b7g-e4` (de4b524) | `open1-b5-b7g-e4` (49119f9e) | open1-b7g-e4 | 68e19e34 | 1.0.0-rc.1, 1.0.0-rc.2, 1.0.0-rc.3, 1.0.0 |
| open1-b5-e4-bialy | `heads-open1-b5-e4-bialy` (71b7752) | `open1-b5-e4-bialy` (39012d07) | open1-b5-e4-bialy | not recorded | none |
| open1-b5-e4 | `heads-open1-b5-e4` (71b7752) | not on the `bialy` Hub repository; published before the rename | open1-b5-e4 | not recorded | none |

### Datasets

| Version | Tag (commit) | Hub tag (revision) | Derived from |
|---|---|---|---|
| open1-b7g-e4 | `dataset-open1-b7g-e4` (de4b524) | `open1-b7g-e4` (48f76bcf) | open1-b5-e4, guide labels relabelled from tool calls |
| open1-b5-e4-bialy | `dataset-open1-b5-e4-bialy` (71b7752) | `open1-b5-e4-bialy` (24918f78) | open1-b5-e4, renamed |
| open1-b5-e4 | `dataset-open1-b5-e4` (71b7752) | not on the `bialy-dataset` Hub repository; published before the rename | |

The open1 releases were made before releases were tagged and before this
repository's history was flattened into "Initial Bialy release". Their tags
point at the first commit on `main` that records their anchors, which carries
the code as it stood when the history was flattened, not necessarily the exact
commit that built them; their provenance does not name a factory commit.
Releases from here on name the commit that built them.
