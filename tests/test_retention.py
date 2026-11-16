import dataclasses

from bialy import retention


def test_shape_archetypes_replaces_only_what_the_run_sets(factory):
    archetypes = factory.archetypes
    assert retention.shape_archetypes(archetypes, {}) == archetypes
    shaped = retention.shape_archetypes(archetypes, {"follow_up_rate": 1.0, "follow_ups": 3})
    assert all(a.follow_up_rate == 1.0 and a.max_follow_ups == 3 for a in shaped)
    assert [dataclasses.replace(a, follow_up_rate=0, max_follow_ups=0) for a in shaped] == [dataclasses.replace(a, follow_up_rate=0, max_follow_ups=0) for a in archetypes]


def test_compare_uses_only_requests_both_passes_drove():
    def row(task, index, tokens, partial=False):
        return {"host": "coordinator", "surface": "s", "partial": partial, "request_index": index, "calls": 3, "prompt_tokens": tokens, "recall_calls": 0,
                "repeated_calls": 1, "meta": {"task_id": task}, "retention": {"asked": True, "dropped": 1}}
    out = retention.compare([row("a", 0, 100), row("b", 0, 300, partial=True)], [row("a", 0, 80), row("b", 0, 10, partial=True), row("c", 0, 5)])
    assert out["requests"] == 1 and out["on_only"] == 1 and out["off_only"] == 0
    assert out["measures"]["prompt_tokens"]["delta"] == -20 and out["measures"]["repeated_calls"]["delta"] == 0 and out["asked"] == 1
    assert retention.compare([], [])["requests"] == 0
