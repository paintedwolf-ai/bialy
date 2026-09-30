"""Discovery includes visibility cutoffs and unknown-label handling."""
import collections
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('rank_eval', Path(__file__).parents[1] / 'train-host/rank_eval.py')
rank_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank_eval)


def row(scores, labels, requests=()):
    return {'skills': {'scores': scores, 'labels': labels}, 'requests': list(requests)}


def test_visibility_requires_score_and_rank():
    rows = [row({'direct': .9, 'other': .8}, {'direct': [4, 4], 'other': [0, 0]})]
    report = rank_eval.metrics(rows, ['direct', 'other'], collections.Counter(), 3.49)
    assert report['any_raw@6'] == 1
    assert report['any_visible@6'] == 0
    assert report['bands']['rare']['support'] == 1
    assert report['no_skill_rows'] == 0


def test_unknown_and_disagreement_are_not_no_skill_evidence():
    rows = [row({'a': 3.6, 'b': 0}, {'b': [0, 0]}),
            row({'a': 3.6, 'b': 0}, {'a': [2, 4], 'b': [0, 0]}),
            row({'a': 3.6, 'b': 0}, {'a': [0, 0], 'b': [0, 0]})]
    report = rank_eval.metrics(rows, ['a', 'b'], collections.Counter())
    assert report['preloads'] == 3
    assert report['preloads_unscored'] == 1
    assert report['preloads_disputed'] == 1
    assert report['preloads_unlikely'] == 1
    assert report['no_skill_rows'] == 1
    assert report['no_skill_preloads'] == 1


def test_request_nearest_and_threshold_use_same_order():
    rows = [row({}, {}, [{'scores': {'a': 2, 'b': 1}, 'wanted': ['b']}])]
    assert rank_eval.request_metrics(rows, 3, 4, 1)['served'] == 0
    assert rank_eval.request_metrics(rows, 3, 4, 2)['served'] == 1


parity_spec = importlib.util.spec_from_file_location('rank_parity', Path(__file__).parents[1] / 'train-host/rank_parity.py')
parity = importlib.util.module_from_spec(parity_spec)
parity_spec.loader.exec_module(parity)


def test_parity_requires_matching_coverage_and_detects_large_flips():
    import pytest

    left = {'one': row({'a': 3.8, 'b': 0}, {})}
    right = {'one': row({'a': 3.0, 'b': 0}, {})}
    assert not parity.compare(left, right, 3.49)['passed']
    with pytest.raises(ValueError, match='row sets differ'):
        parity.compare(left, {}, 3.49)
    assert parity.compare(left, left, 3.49)['passed']


def test_parity_includes_request_decisions():
    left = {'one': row({'a': 3.8}, {}, [{'scores': {'x': 3, 'y': 0}, 'wanted': ['x']}])}
    right = {'one': row({'a': 3.8}, {}, [{'scores': {'x': 0, 'y': 3}, 'wanted': ['x']}])}
    result = parity.compare(left, right, 3.49)
    assert result['request_flips'] > 0
    assert result['request_far_flips'] > 0
    assert not result['passed']


def test_prediction_head_identity_changes_when_weights_change(tmp_path):
    import pytest

    head = tmp_path / 'head.safetensors'
    head.write_bytes(b'first weights')
    first = rank_eval.head_hashes(['unit-rank=' + str(head)])
    head.write_bytes(b'replaced weights with the same display label')
    assert first != rank_eval.head_hashes(['unit-rank=' + str(head)])
    with pytest.raises(ValueError, match='require'):
        rank_eval.head_hashes(None)
    with pytest.raises(ValueError, match='duplicate'):
        rank_eval.head_hashes(['unit-rank=' + str(head)] * 2)


def test_listing_calibration_does_not_change_preloads():
    rows = [row({'direct': .8, 'other': .2}, {'direct': [4, 4], 'other': [0, 0]})]
    report = rank_eval.metrics(rows, ['direct', 'other'], collections.Counter(), 3.9, .5)
    assert report['any_visible@6'] == 1
    assert report.get('preloads', 0) == 0
    left = {'one': row({'direct': .8}, {})}
    right = {'one': row({'direct': .2}, {})}
    assert parity.compare(left, right, 3.9)['passed']
    compared = parity.compare(left, right, 3.9, list_at=.5)
    assert not compared['passed']
    assert compared['visible_top6_set_changes'] == 1
    assert compared['far_flips'] == 1


def test_parity_retains_strict_failure_for_hidden_score_crossing():
    scores = {str(i): 4 - i * .2 for i in range(8)}
    left = {'one': row({**scores, 'hidden': .4}, {})}
    right = {'one': row({**scores, 'hidden': .9}, {})}
    result = parity.compare(left, right, 3.99, list_at=.5)
    assert not result['passed']
    assert result['far_flips'] == result['hidden_far_threshold_flips'] == 1
    assert result['visible_top6_set_changes'] == 0
    assert result['visible_top8_set_changes'] == 0
    assert result['preload_identity_changes'] == 0


def test_request_surface_includes_exact_names_and_negative_requests():
    rows = [row({}, {}, [
        {'scores': {'other': .2}, 'wanted': [], 'exact': ['command'], 'called': ['command'],
         'labels': {'command': [4, 4], 'other': [0, 0]}},
        {'scores': {'write': 3}, 'wanted': ['write'], 'exact': [], 'called': ['write'],
         'labels': {'write': [3, 4]}},
        {'scores': {}, 'wanted': [], 'exact': ['read'], 'called': ['read']},
    ])]
    ranked = rank_eval.request_metrics(rows)
    assert ranked['needs'] == 1
    assert ranked['served'] == 1
    report = rank_eval.request_surface_metrics(rows)
    assert report['requests'] == 3
    assert report['all_called_served_rate'] == 1
    assert report['selected_tools'] == 4
    assert report['judged_precision'] == 2 / 3
