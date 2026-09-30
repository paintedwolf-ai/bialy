import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('tool_event_rows', Path(__file__).parents[1] / 'train-host/tool_event_rows.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_event_variants_use_recorded_loadable_calls_and_preserve_labels():
    row = {'session': 's', 'receipt': 7, 'partial': False, 'state': {'user': '  ' + 'é' * 902 + '  '},
           'offered': {'loadable': ['command', 'edit']},
           'labels': {'tools': ['read', 'command', 'command', 'edit'], 'skill_scores': {'skill': 3}, 'requests': [{}]}}
    got = list(module.variants([row, {**row, 'partial': True}]))
    assert len(got) == 2
    assert got[0]['state']['user'] == 'é' * 900 + '\n\nThe agent just called `command`.'
    assert got[0]['labels']['skill_scores'] == row['labels']['skill_scores']
    assert got[0]['labels']['requests'] == []
    assert row['labels']['requests'] == [{}]
    assert got[0]['session'] != got[1]['session']
    assert got == list(module.variants([row]))
