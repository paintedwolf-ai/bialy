import dataclasses

from bialy import retention


def test_shape_archetypes_replaces_only_what_the_run_sets(factory):
    archetypes = factory.archetypes
    assert retention.shape_archetypes(archetypes, {}) == archetypes
    shaped = retention.shape_archetypes(archetypes, {"follow_up_rate": 1.0, "follow_ups": 3})
    assert all(a.follow_up_rate == 1.0 and a.max_follow_ups == 3 for a in shaped)
    assert [dataclasses.replace(a, follow_up_rate=0, max_follow_ups=0) for a in shaped] == [dataclasses.replace(a, follow_up_rate=0, max_follow_ups=0) for a in archetypes]
