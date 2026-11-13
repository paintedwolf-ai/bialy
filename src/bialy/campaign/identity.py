"""The separate identities a campaign artifact is bound to.

Each kind names one thing that can change on its own: the host that captured sessions,
the factory, the trainer, the converter, the serving runtime, a training recipe, a
student model and its chat template, the scenario bank, and a sealed dataset. An
artifact binds only the kinds it depends on, so a trainer fix leaves captures valid
while every trained head or student it produced is bound to the trainer that made it.
"""

from .journal import CampaignFault, digest

KINDS = ("capture_host", "factory", "trainer", "converter", "server", "recipe", "model_template", "scenario", "dataset")

# What each artifact depends on.
DEPENDS = {
    "capture": ("capture_host", "factory", "scenario", "model_template"),
    "dataset": ("capture_host", "factory", "scenario", "model_template", "dataset"),
    "training": ("capture_host", "factory", "scenario", "model_template", "dataset", "trainer", "recipe"),
    "export": ("capture_host", "factory", "scenario", "model_template", "dataset", "trainer", "recipe", "converter"),
    "evaluation": ("capture_host", "factory", "scenario", "model_template", "dataset", "trainer", "recipe", "converter", "server"),
}


class Identities:
    """One value per identity kind, each a commit, digest or pinned revision."""

    def __init__(self, values):
        unknown = set(values) - set(KINDS)
        if unknown:
            raise CampaignFault("identity_kind_unknown", details={"kinds": sorted(unknown)})
        empty = sorted(kind for kind, value in values.items() if not value or not isinstance(value, str))
        if empty:
            raise CampaignFault("identity_value_invalid", details={"kinds": empty})
        self.values = dict(sorted(values.items()))

    def binding(self, artifact):
        """The identities `artifact` depends on and their digest; every one must be known."""
        try:
            kinds = DEPENDS[artifact]
        except KeyError:
            raise CampaignFault("identity_artifact_unknown", details={"artifact": artifact}) from None
        missing = [kind for kind in kinds if kind not in self.values]
        if missing:
            raise CampaignFault("identity_missing", details={"artifact": artifact, "kinds": missing})
        bound = {kind: self.values[kind] for kind in kinds}
        return {"artifact": artifact, "identities": bound, "sha256": digest(bound)}

    def still_valid(self, artifact, binding):
        """Whether an artifact bound earlier still matches these identities."""
        return binding.get("artifact") == artifact and self.binding(artifact)["sha256"] == binding.get("sha256")
