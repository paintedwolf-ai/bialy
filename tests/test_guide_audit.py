import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('guide_audit', Path(__file__).parents[1] / 'train-host/guide_audit.py')
guide_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guide_audit)


def test_unlabeled_and_wrongly_omitted_guides_are_not_certified():
    rows = []
    for n in range(20):
        needed = n >= 10
        rows.append({'labels': {'guides': {'supported': needed, 'unknown': None, 'wrong': needed}},
                     'answers': {'guides': {'probabilities': {'supported': .9 if needed else .1, 'unknown': .1, 'wrong': .1}}}})
    result = guide_audit.audit(rows, rows)
    assert result['omittable'] == ['supported']
    assert result['validation']['unknown']['omitted_unknown'] == 20
    assert result['validation']['wrong']['omitted_needed'] == 10


def test_high_precision_cannot_hide_lost_rare_needed_guides():
    rows = [{'labels': {'guides': {'rare': n >= 195}},
             'answers': {'guides': {'probabilities': {'rare': .01}}}} for n in range(200)]
    result = guide_audit.audit(rows, rows)
    assert result['omittable'] == []
    assert result['validation']['rare']['omitted_needed'] == 5
