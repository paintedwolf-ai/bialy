import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('request_expansion_eval', Path(__file__).parents[1] / 'train-host/request_expansion_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_expansion_keeps_selection_fixed_and_respects_surface():
    catalog = {'families': {'command': ['command', 'command_output', 'command_stop']},
               'request_companions': {'command': ['git_checkout', 'http_request']}}
    offered = {'one': {'loadable': ['command', 'command_output', 'git_checkout'], 'floor': []}}
    predictions = [{'key': 'one', 'requests': [{'scores': {'command': 3.5}, 'exact': [], 'called': ['command']}]}]
    result = module.compare(predictions, offered, catalog, 2, 4, 1)
    sets = result['details'][0]['sets']
    assert sets['none'] == ['command']
    assert sets['families'] == ['command', 'command_output']
    assert sets['all'] == ['command', 'command_output', 'git_checkout']
    assert all(v['all_called_rate'] == 1 for v in result['summary'].values())
