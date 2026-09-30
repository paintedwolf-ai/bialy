import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'train-host'))
from turn_parity import compare  # noqa: E402


def predictions(value):
    return {'row': {'answers': {'tools': {'probabilities': {'command': value}}}}}


def test_turn_parity_rejects_empty_or_different_coverage_and_far_flips():
    with pytest.raises(ValueError, match='coverage'):
        compare({}, {}, .89)
    with pytest.raises(ValueError, match='coverage'):
        compare(predictions(.9), {}, .89)
    assert compare(predictions(.9), predictions(.91), .89)['passed']
    assert not compare(predictions(.9), predictions(.2), .89)['passed']
    assert compare(predictions(.9), predictions(.2), .89)['far_flips'] == 1
