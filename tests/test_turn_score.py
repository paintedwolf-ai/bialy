import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('turn_score', Path(__file__).parents[1] / 'train-host/turn_score.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unknown_labels_do_not_become_negative_but_still_count_as_loads():
    predictions = [{'key': 'one', 'answers': {'tools': {'probabilities': {'a': .9, 'b': .8, 'c': .7}}}}]
    result = module.score(predictions, {'one': {}}, lambda _: {'a': 1, 'b': None, 'c': 0}, .75)
    assert result['counts']['tp'] == 1
    assert result['counts'].get('fp', 0) == 0
    assert result['counts']['unknown_loaded'] == 1
    assert result['loads_per_turn'] == 2
    assert result['precision'] == 1


def test_removed_catalog_tools_are_reported_separately():
    predictions = [{'key': 'one', 'answers': {'tools': {'probabilities': {'a': .9}}}}]
    result = module.score(predictions, {'one': {}}, lambda _: {'a': 1, 'fixture': 1}, .75, {'a'})
    assert result['recall'] == 1
    assert result['counts']['out_of_catalog_labels'] == 1
    assert result['counts']['out_of_catalog_positives'] == 1
