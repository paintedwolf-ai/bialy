import pytest

from bialy.campaign.journal import CampaignFault
from bialy.campaign.oracle import seed
from bialy.campaign.pilot_bank import build, examples


def test_bank_has_frozen_disjoint_partitions_and_seeds_exact_sources(tmp_path):
    bank = tmp_path / 'bank'
    tasks = build(bank)
    assert len(tasks) == 32
    assert len({task['id'] for task in tasks}) == 32
    assert build(bank) == tasks
    for task in tasks:
        destination = tmp_path / 'projects' / task['id']
        seed(task, bank / task['oracle_directory'], destination)
        for name, source in task['fixture_files'].items():
            assert (destination / name).read_bytes() == (bank / task['oracle_directory'] / source).read_bytes()
        compile((bank / task['oracle_directory'] / 'check.py').read_text(), 'oracle', 'exec')
    task = next(task for task in tasks if task['fixture_files'])
    with pytest.raises(CampaignFault, match='fixture_destination_exists'):
        seed(task, bank / task['oracle_directory'], tmp_path / 'projects' / task['id'])
    source = next(iter(task['fixture_files'].values()))
    (bank / task['oracle_directory'] / source).write_text('changed')
    with pytest.raises(CampaignFault, match='fixture_source_changed'):
        seed(task, bank / task['oracle_directory'], tmp_path / 'changed')


@pytest.mark.parametrize('name', ['../outside', '/absolute', '.git/config'])
def test_fixture_paths_cannot_escape_project(tmp_path, name):
    with pytest.raises(CampaignFault, match='fixture_path_outside_project'):
        seed({'fixture_files': {name: 'source'}}, tmp_path, tmp_path / 'project')

def test_missing_command_oracle_checks_the_calculation_beyond_the_visible_example(tmp_path, monkeypatch):
    import sys
    import types

    _, _, assertions = examples(46)["missing_path_or_executable"]
    module = types.ModuleType("value")
    monkeypatch.setitem(sys.modules, "value", module)
    module.compute = lambda n: 46
    with pytest.raises(AssertionError):
        exec(assertions)
    module.compute = lambda n: n + 43
    exec(assertions)
