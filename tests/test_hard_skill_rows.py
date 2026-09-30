import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('hard_skill_rows', Path(__file__).parents[1] / 'train-host/hard_skill_rows.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sampling_preserves_nonzero_pairs_and_prioritizes_hard_negatives():
    scores = {'right': 4, 'maybe': 2, 'easy': 0, 'hard': 0, 'other': 0}
    row = {'session': 's', 'receipt': 1, 'partial': False,
           'labels': {'skill_scores': scores, 'second_scores': {'skills': scores}, 'requests': [{'need': 'unchanged'}]}}
    pred = {'skills': {'scores': {'right': 3, 'maybe': 1, 'easy': .1, 'hard': 3.9, 'other': .2},
                       'labels': {name: [value, value] for name, value in scores.items()}}}
    out = module.sample(row, pred, hard=1, random_zeros=1)
    assert {'right', 'maybe', 'hard'} <= set(out['labels']['skill_scores'])
    assert len(out['labels']['skill_scores']) == 4
    assert out['labels']['skill_scores']['hard'] == 0
    assert out['labels']['requests'] == row['labels']['requests']
    assert len(row['labels']['skill_scores']) == 5
    assert module.sample(row, pred, hard=1, random_zeros=1) == out
    pred['skills']['labels']['hard'] = [0, 4]
    with pytest.raises(ValueError, match='labels differ'):
        module.sample(row, pred)
