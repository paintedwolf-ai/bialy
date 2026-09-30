import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'train-host'))
from turn_audit import gaps, variants  # noqa: E402


def test_variants_do_not_mutate_the_baseline_or_each_other():
    request = {'method': 'decide', 'questions': {'tools': {'options': {'command': 'run'}},
                                               'kind': {'options': {'run': 'execute'}}}}
    original = copy.deepcopy(request)
    generated = dict(variants(request, 'extra', 'unrelated'))
    assert request == original
    assert len(generated['tools_only']['questions']) == 1
    assert 'extra' not in generated['repeat']['questions']['tools']['options']
    assert 'extra' in generated['added_tool']['questions']['tools']['options']
    with pytest.raises(ValueError, match='already exists'):
        variants(request, 'command', 'run')


def test_roster_comparison_includes_all_old_tools_but_not_the_new_one():
    assert gaps({'command': .8}, {'command': .3, 'extra': .9}) == {'command': .5}
    with pytest.raises(ValueError, match='missing'):
        gaps({'command': .8}, {'extra': .8})
    with pytest.raises(ValueError, match='missing'):
        gaps({}, {})


def test_independent_corpus_keeps_loading_contract_and_disables_untrained_questions():
    from independent_corpus import prepare

    source = {'questions': {'tools': {'option_words': 6, 'options': {'old': 'Previous release'}},
                            'guides': {'omittable': ['read'], 'omit_below': .3},
                            'kind': {'veto_tools': True}},
              'tools': [{'name': 'command', 'description': 'Run a command',
                         'request_loads': ['command', 'command_output', 'command_stop']}]}
    result = prepare(source)
    assert result['questions']['tools']['independent']
    assert 'options' not in result['questions']['tools']
    assert result['questions']['guides']['omittable'] == []
    assert not result['questions']['kind']['veto_tools']
    assert result['tools'][0]['request_loads'] == source['tools'][0]['request_loads']
    assert result['tools'][0]['option'] == 'Run a command'
    assert source['questions']['tools']['option_words'] == 6
