import json
import os
import subprocess
import sys

import pytest
import yaml

from bialy.campaign import ancestry, scenarios
from bialy.campaign.journal import CampaignFault

CATALOG = scenarios.load()
CASES = [(group, index) for group in CATALOG.groups for index in range(len(group.variants))]


def evidence(root, verdict='passed', interactive=True, reproduced=True):
    """A host bundle's operations: optionally a failed check of the original source r0, an edit that left
    revision r1, optionally a check of r1, optionally a terminal answer."""
    bundle = root / 'host-bundle'
    bundle.mkdir(parents=True)
    base = {'session_id': 'root', 'source': 'invocation-receipt', 'contract': 'pw-operation-receipt/1', 'invoked': True,
            'settlement': 'settled', 'semantic_outcome': 'success', 'effect_state': 'known'}
    operations = [dict(base, id='first-check', tool='command', source_revision='r0', source_verdict='failed',
                       started_at='2026-11-12T09:59:00Z')] if reproduced else []
    operations.append(dict(base, id='edit', tool='edit', source_revision='r1', started_at='2026-11-12T10:00:00Z'))
    if verdict:
        operations.append(dict(base, id='check', tool='command', source_revision='r1', source_verdict=verdict,
                               started_at='2026-11-12T10:00:01.5Z'))
    if interactive:
        operations.append(dict(base, id='answer', tool='terminal_send', started_at='2026-11-12T10:00:02Z'))
    (bundle / 'operations.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in operations))
    return root


def grade(tmp_path, group, index, changes, host, program='check.py'):
    """Run the task's grader on the fixtures with `changes` applied, as oracle.grade does."""
    oracle = tmp_path / 'bank'
    task = next(task for task in scenarios.tasks(CATALOG, oracle, lambda name: '/work/' + name)
                if task['id'] == '%s-%d' % (group.id, index))
    work = tmp_path / ('work-' + os.urandom(4).hex())
    work.mkdir()
    parameters = group.variants[index]
    for name, body in {**group.files(group.fixtures, parameters), **group.files(changes, parameters)}.items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_text(body)
    result = subprocess.run([sys.executable, str(oracle / task['oracle_directory'] / program)], cwd=work,
                            env=dict(os.environ, BIALY_GRADER_EVIDENCE=str(host)), capture_output=True, text=True, timeout=60)
    return result.returncode == 0, result.stderr


@pytest.mark.parametrize('group,index', CASES, ids=['%s-%d' % (group.id, index) for group, index in CASES])
def test_every_grader_passes_only_the_verified_solution(tmp_path, group, index):
    verified, plain, unverified = evidence(tmp_path / 'verified'), evidence(tmp_path / 'plain', interactive=False), \
        evidence(tmp_path / 'unverified', verdict=None)
    passed, detail = grade(tmp_path, group, index, group.solution, verified)
    assert passed, detail
    assert not grade(tmp_path, group, index, {}, verified)[0], 'the unchanged fixtures pass'
    if group.verification:
        assert not grade(tmp_path, group, index, group.solution, unverified)[0], 'a solution without verification passes'
        failed_check = evidence(tmp_path / 'failed', verdict='failed')
        assert not grade(tmp_path, group, index, group.solution, failed_check)[0], 'a failed verification counts'
    assert group.superficial or group.tampered, 'no manufactured-pass control'
    if group.superficial:
        assert not grade(tmp_path, group, index, group.superficial, plain)[0], 'a superficial change passes'
    if group.tampered:
        assert not grade(tmp_path, group, index, group.tampered, verified)[0], 'tampered tests pass'
    if group.diagnostic:
        assert grade(tmp_path, group, index, group.solution, verified, 'diagnostic.py')[0], 'a reproduced failure is not witnessed'
        unreproduced = evidence(tmp_path / 'unreproduced', reproduced=False)
        assert not grade(tmp_path, group, index, group.solution, unreproduced, 'diagnostic.py')[0], 'an unreproduced failure is witnessed'


def test_all_eight_focuses_load_with_groups_for_the_pilot():
    assert set(CATALOG.focuses) == {'implementation_investigation', 'failure_diagnosis', 'interactive_work', 'build_test_repair',
                                    'partial_effect_recovery', 'capability_recovery', 'sustained_work', 'evidence_closeout'}
    assert scenarios.deficits(CATALOG, 'pilot') == {}


def test_a_short_focus_reports_its_exact_deficit():
    need = sum(CATALOG.quotas['full']['minimums'].values())
    assert scenarios.deficits(CATALOG, 'full') == {focus: need - len(groups) for focus, groups in CATALOG.by_focus().items()}


def test_the_pilot_allocation_gives_each_focus_one_group_per_partition_and_keeps_variants_together(tmp_path):
    task_list = scenarios.tasks(CATALOG, tmp_path / 'bank', lambda name: '/work/' + name)
    assignment = scenarios.assignment(CATALOG, task_list, 'seed', 'pilot')
    by_group = {}
    for task in task_list:
        by_group.setdefault(task['group'], set()).add(assignment.partition(ancestry.task_node(task['id'])))
    assert all(len(partitions) == 1 for partitions in by_group.values())
    for focus in CATALOG.focuses:
        partitions = sorted(next(iter(by_group[group.id])) for group in CATALOG.groups if group.focus == focus)
        assert partitions == sorted(ancestry.PARTITIONS)
    with pytest.raises(CampaignFault, match='coverage_short') as short:
        scenarios.assignment(CATALOG, task_list, 'seed', 'full')
    assert short.value.details['deficits'] == scenarios.deficits(CATALOG, 'full')


def write(tmp_path, groups, quotas=None):
    value = yaml.safe_load(scenarios.CATALOG.read_text())
    value['groups'] = groups
    if quotas:
        value['quotas'].update(quotas)
    path = tmp_path / 'catalog.yaml'
    path.write_text(yaml.safe_dump(value))
    return path


def test_the_catalog_refuses_malformed_groups(tmp_path):
    raw = yaml.safe_load(scenarios.CATALOG.read_text())['groups'][0]
    with pytest.raises(CampaignFault, match='scenario_variant_quota'):
        scenarios.load(write(tmp_path, [dict(raw, variants=[{'threshold': n} for n in range(5)])]))
    with pytest.raises(CampaignFault, match='scenario_variant_quota'):
        scenarios.load(write(tmp_path, [dict(raw, variants=[{'threshold': 3}, {'threshold': 3}])]))
    with pytest.raises(CampaignFault, match='scenario_group_invalid'):
        scenarios.load(write(tmp_path, [dict(raw, tool='edit')]))
    with pytest.raises(CampaignFault, match='scenario_group_invalid'):
        scenarios.load(write(tmp_path, [dict(raw, focus='unknown')]))
    with pytest.raises(CampaignFault, match='scenario_group_invalid'):
        scenarios.load(write(tmp_path, [raw, raw]))
    catalog = scenarios.load(write(tmp_path, [dict(raw, variants=[{'other': 1}])]))
    with pytest.raises(CampaignFault, match='scenario_template_invalid'):
        scenarios.tasks(catalog, tmp_path / 'bank', lambda name: '/work/' + name)


def test_two_groups_with_one_prompt_cannot_cross_a_split(tmp_path):
    raw = yaml.safe_load(scenarios.CATALOG.read_text())['groups'][0]
    catalog = scenarios.load(write(tmp_path, [raw, dict(raw, id='copy', solution={'other.py': 'x'})]))
    task_list = scenarios.tasks(catalog, tmp_path / 'bank', lambda name: '/work/' + name)
    with pytest.raises(CampaignFault, match='duplicate_across_components'):
        scenarios.assignment(catalog, task_list, 'seed', 'pilot')


def test_a_changed_bank_is_refused(tmp_path):
    scenarios.tasks(CATALOG, tmp_path / 'bank', lambda name: '/work/' + name)
    first = CATALOG.groups[0]
    (tmp_path / 'bank' / (first.id + '-0') / 'check.py').write_text('raise SystemExit(0)\n')
    with pytest.raises(CampaignFault, match='scenario_bank_changed'):
        scenarios.tasks(CATALOG, tmp_path / 'bank', lambda name: '/work/' + name)


def test_diagnostic_groups_register_a_required_witness(tmp_path):
    task_list = scenarios.tasks(CATALOG, tmp_path / 'bank', lambda name: '/work/' + name)
    diagnostic = [task for task in task_list if 'diagnostic' in task]
    assert diagnostic and all(task['diagnostic'] == {'expectation': 'reproduced_failure', 'witness': 'diagnostic'}
                              and 'diagnostic' in task['required_checks'] for task in diagnostic)
    raw = yaml.safe_load(scenarios.CATALOG.read_text())['groups'][0]
    with pytest.raises(CampaignFault, match='scenario_group_invalid'):
        scenarios.load(write(tmp_path, [dict(raw, diagnostic={'expectation': 'guessed'})]))
