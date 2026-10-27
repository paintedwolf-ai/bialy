import copy

import pytest

from bialy.campaign.grading import DIMENSIONS, behavioral_result, episode, teacher_routes, wilson
from bialy.campaign.journal import CampaignFault, digest
from bialy.campaign.lineage import FAMILIES


def evidence():
    task = {"required_checks": ["test", "artifact"]}
    receipt = {"task_spec_sha256": digest(task), "grader_identity": "trusted", "complete": True,
               "final_source_sha256": "source", "checks": [{"id": name, "source_sha256": "source", "passed": True}
                                                         for name in task["required_checks"]],
               "evidence_ids": ["check-1"], "human_interventions": 0, "unauthorized_effects": 0,
               "unresolved_effects": 0, "teacher_family": "teacher"}
    reviews = [{"reviewer_slot": name, "judge_id": name, "judge_family": name, "controls_passed": True,
                "episode_sha256": digest(receipt), "evidence_ids": ["check-1"],
                "dimensions": dict.fromkeys(DIMENSIONS, "pass")} for name in ("judge-a", "judge-b")]
    return task, receipt, reviews


def test_false_done_and_failed_tests_cannot_supply_positive_episodes():
    task, receipt, reviews = evidence()
    assert episode(task, receipt, reviews, "trusted")["verdict"] == "pass"
    reviews[0]["dimensions"]["truthful_closeout"] = "fail"
    assert episode(task, receipt, reviews, "trusted")["verdict"] == "fail"
    task, receipt, reviews = evidence()
    receipt["checks"][0]["passed"] = False
    for review in reviews:
        review["episode_sha256"] = digest(receipt)
    assert episode(task, receipt, reviews, "trusted")["verdict"] == "fail"


def test_changed_source_and_unqualified_judge_fail_closed():
    task, receipt, reviews = evidence()
    receipt["final_source_sha256"] = "changed"
    with pytest.raises(CampaignFault, match="source_mismatch"):
        episode(task, receipt, reviews, "trusted")
    task, receipt, reviews = evidence()
    reviews[0]["controls_passed"] = False
    with pytest.raises(CampaignFault, match="review_unqualified"):
        episode(task, receipt, reviews, "trusted")


def test_teacher_floors_follow_measured_yield_without_arbitrary_success_gate():
    measurements = [{"family": family, "teacher_id": "teacher", "qualified": 1, "attempts": 8, "capture_qualified": True}
                    for family in FAMILIES]
    routes = teacher_routes(measurements)
    assert all(row["floor"] == wilson(1, 8)[0] for row in routes["families"].values())
    measurements[0]["qualified"] = 0
    with pytest.raises(CampaignFault, match="insufficient_qualified_data"):
        teacher_routes(measurements)


def evaluation():
    plan = {"tasks": [{"id": family + str(index), "family": family} for family in FAMILIES for index in range(3)], "attempts_per_task": 2,
            "bootstrap_seed": 7, "bootstrap_samples": 1000}
    attempts = [{"task_id": task["id"], "attempt": repetition, "arm": arm, "student": "e4b",
                 "plan_sha256": digest(plan), "effective_thinking": False, "success": arm == "trained",
                 "human_interventions": 0, "unauthorized_effects": 0, "unsupported_completion_claims": 0}
                for task in plan["tasks"] for repetition in range(2) for arm in ("base", "trained")]
    return plan, attempts


def test_paired_evaluation_counts_every_attempt_and_rejects_missing_or_duplicate():
    plan, attempts = evaluation()
    result = behavioral_result(plan, attempts, "e4b")
    assert result["status"] == "pass" and result["paired_task_interval_95"] == [1, 1]
    with pytest.raises(CampaignFault, match="incomplete_or_duplicate"):
        behavioral_result(plan, attempts[:-1], "e4b")
    with pytest.raises(CampaignFault, match="incomplete_or_duplicate"):
        behavioral_result(plan, attempts + [attempts[0]], "e4b")
    changed = copy.deepcopy(attempts)
    changed[-1]["unsupported_completion_claims"] = 1
    assert behavioral_result(plan, changed, "e4b")["status"] == "behavioral_failure"
    changed[-1]["effective_thinking"] = True
    with pytest.raises(CampaignFault, match="thinking_enabled"):
        behavioral_result(plan, changed, "e4b")


def test_duplicate_tasks_cannot_inflate_independent_evaluation_evidence():
    plan, attempts = evaluation()
    plan['tasks'].append(plan['tasks'][0])
    with pytest.raises(CampaignFault, match='invalid_evaluation_plan'):
        behavioral_result(plan, attempts, 'e4b')


def test_only_declared_students_can_be_graded():
    plan, attempts = evaluation()
    with pytest.raises(CampaignFault, match='unknown_student'):
        behavioral_result(plan, attempts, 'undeclared')


def test_teacher_repeated_measurement_cannot_select_a_lucky_attempt():
    measurements = [{'family': family, 'teacher_id': 'teacher', 'qualified': 1, 'attempts': 8, 'capture_qualified': True}
                    for family in FAMILIES]
    with pytest.raises(CampaignFault, match='duplicate_teacher_measurement'):
        teacher_routes(measurements + [measurements[0]])


def test_single_qualified_same_family_review_can_admit_an_episode():
    task, receipt, reviews = evidence()
    reviews[0]['judge_family'] = receipt['teacher_family']
    assert episode(task, receipt, reviews[:1], 'trusted')['verdict'] == 'pass'
