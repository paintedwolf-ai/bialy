import dataclasses

from bialy import retention


def test_search_grid_names_every_point_in_a_fixed_order():
    grid = retention.search_grid({"search": {"lr": [1e-3, 5e-4], "pos_weight": [6.0], "truth": ["needed", "later"]}})
    assert [name for name, _ in grid] == ["lr0.001-pw6-needed", "lr0.001-pw6-later", "lr0.0005-pw6-needed", "lr0.0005-pw6-later"]
    assert grid[1][1] == ["--lr", "0.001", "--pos-weight", "6.0", "--retention-truth", "later"]
    assert retention.search_grid({}) == [("lr0.0005-pw6-needed", ["--lr", "0.0005", "--pos-weight", "6.0", "--retention-truth", "needed"])]


def test_select_takes_the_best_validation_objective_and_the_stricter_tie():
    reports = {"a": {"fitted": {"tally": {"objective_tokens": 100, "released": 9}}},
               "b": {"fitted": {"tally": {"objective_tokens": 300, "released": 20}}},
               "c": {"fitted": {"tally": {"objective_tokens": 300, "released": 12}}}}
    assert retention.select(reports) == ("c", 300)
    assert retention.select({}) == (None, None)


def test_shape_archetypes_replaces_only_what_the_run_sets(factory):
    archetypes = factory.archetypes
    assert retention.shape_archetypes(archetypes, {}) == archetypes
    shaped = retention.shape_archetypes(archetypes, {"follow_up_rate": 1.0, "follow_ups": 3})
    assert all(a.follow_up_rate == 1.0 and a.max_follow_ups == 3 for a in shaped)
    assert [dataclasses.replace(a, follow_up_rate=0, max_follow_ups=0) for a in shaped] == [dataclasses.replace(a, follow_up_rate=0, max_follow_ups=0) for a in archetypes]


def held(low, objective=5000, best_fact=100, rows=50, abstained=0):
    return {"holdout": {"rows": rows, "verified_label_rows": rows, "auc": 0.8, "needed_rate": 0.2, "abstained": abstained,
                        "engine": {"candidates": 400, "objective_tokens": objective, "objective_per_request": objective // 50, "objective_share": 0.1, "released": 20, "needed_released": 0},
                        "structural": {"none": {"objective_tokens": 0}, "repeated": {"objective_tokens": -300}, "stale": {"objective_tokens": best_fact}},
                        "bootstrap": {"engine": {"low": low, "median": 100, "high": 200, "draws": 500}}, "policy": {"drop_below": 0.2, "confidence_floor": 0.7}},
            "recommendation": {"facts": ["stale"] if best_fact > 0 else []}}


def test_viability_reads_the_interval_the_structural_facts_and_the_comparison():
    good = retention.viability(held(low=10), None, {"min_rows": 20})
    assert good["viable"] and good["numbers"]["facts"] == [] and good["numbers"]["interval"]["low"] == 10
    for report, reason in ((held(low=-5), "not clear of releasing nothing"), (held(low=10, objective=0), "does not beat the deterministic baseline"),
                           (held(low=10, rows=5), "an interval needs 20"), (held(low=10, abstained=3), "did not load")):
        verdict = retention.viability(report, None, {"min_rows": 20})
        assert not verdict["viable"] and any(reason in r for r in verdict["reasons"]), verdict
    comparison = {"requests": 40, "released_results": 12, "measures": {"prompt_tokens": {"off": 10, "on": 8, "delta": -2, "wants": "down"},
                                                                       "recall_calls": {"off": 0.1, "on": 0.3, "delta": 0.2, "wants": "up"}}}
    verdict = retention.viability(held(low=10), comparison, {"min_rows": 20})
    assert verdict["viable"]
    comparison["measures"]["recall_calls"]["delta"] = 0.0
    assert retention.viability(held(low=10), comparison, {"min_rows": 20})["viable"]
    comparison["measures"]["prompt_tokens"]["delta"] = 1
    assert "prompt_tokens did not fall" in retention.viability(held(low=10), comparison, {"min_rows": 20})["reasons"][0]
    assert not retention.viability({}, None, {})["viable"]


def test_compare_uses_only_requests_both_passes_drove():
    def row(task, index, tokens, partial=False):
        return {"host": "coordinator", "surface": "s", "partial": partial, "request_index": index, "calls": 3, "prompt_tokens": tokens, "recall_calls": 0,
                "repeated_calls": 1, "meta": {"task_id": task}, "retention": {"asked": True, "dropped": 1}}
    out = retention.compare([row("a", 0, 100), row("b", 0, 300, partial=True)], [row("a", 0, 80), row("b", 0, 10, partial=True), row("c", 0, 5)])
    assert out["requests"] == 1 and out["on_only"] == 1 and out["off_only"] == 0
    assert out["measures"]["prompt_tokens"]["delta"] == -20 and out["measures"]["repeated_calls"]["delta"] == 0 and out["asked"] == 1
    assert retention.compare([], [])["requests"] == 0


def test_needed_release_requires_verified_recovery():
    replay = held(low=10)
    replay["holdout"]["engine"]["needed_released"] = 1
    verdict = retention.viability(replay, None, {})
    assert not verdict["viable"]
    assert "without verified recovery" in verdict["reasons"][0]


def test_unverified_hindsight_cannot_qualify_a_head():
    replay = held(low=10)
    del replay["holdout"]["verified_label_rows"]
    verdict = retention.viability(replay, None, {})
    assert not verdict["viable"]
    assert "verified task or judge provenance" in verdict["reasons"][0]
