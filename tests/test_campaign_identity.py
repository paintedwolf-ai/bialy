import pytest

from bialy.campaign.identity import DEPENDS, KINDS, Identities
from bialy.campaign.journal import CampaignFault

VALUES = {kind: "%s-1" % kind for kind in KINDS}


def test_a_trainer_change_keeps_captures_and_rebinds_what_it_trains():
    before, after = Identities(VALUES), Identities(dict(VALUES, trainer="trainer-2"))
    capture, training = before.binding("capture"), before.binding("training")
    assert after.still_valid("capture", capture) and after.still_valid("dataset", before.binding("dataset"))
    assert not after.still_valid("training", training)
    assert not after.still_valid("evaluation", before.binding("evaluation"))


def test_a_scenario_change_invalidates_everything_built_on_it():
    before, after = Identities(VALUES), Identities(dict(VALUES, scenario="scenario-2"))
    assert all(not after.still_valid(artifact, before.binding(artifact)) for artifact in DEPENDS)


def test_a_binding_names_only_the_identities_its_artifact_depends_on():
    binding = Identities(VALUES).binding("capture")
    assert set(binding["identities"]) == set(DEPENDS["capture"]) and "trainer" not in binding["identities"]


def test_unknown_missing_and_empty_identities_are_refused():
    with pytest.raises(CampaignFault, match="identity_kind_unknown"):
        Identities(dict(VALUES, hostname="x"))
    with pytest.raises(CampaignFault, match="identity_value_invalid"):
        Identities(dict(VALUES, trainer=""))
    with pytest.raises(CampaignFault, match="identity_missing"):
        Identities({"capture_host": "h"}).binding("capture")
    with pytest.raises(CampaignFault, match="identity_artifact_unknown"):
        Identities(VALUES).binding("poster")
