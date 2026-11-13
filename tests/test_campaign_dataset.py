import json

import pytest
from conftest import FOCUSES, register_bank
from host_bundles import write_episode

from bialy.campaign import coverage
from bialy.campaign.ancestry import PARTITIONS
from bialy.campaign.dataset import MAX_TOKENS, assemble
from bialy.campaign.episodes import Episodes
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal, digest
from bialy.campaign.lineage import Lineage
from bialy.provenance import file_sha256
from bialy.student_host import measure_rows
from bialy.student_host.specification import load, students

P0 = {'pilot': '/pilot', 'policy': 'p0', 'fallback_rate': 0.1}
MINIMUM = {'train': 8, 'validation': 8, 'system_test': 8}
COMPACTED = [{'role': 'user', 'content_parts': [{'source': 'compaction', 'origin': 'host'}]}]


def row(session, bundle, category='ordinary', fallback=False, catalog='k' * 64):
    capture = {'schema': 'pw-model-request-capture/3', 'session_id': session, 'catalog_sha256': catalog, 'tools': []}
    capture['sha256'] = digest(capture)
    target = {'role': 'assistant', 'text': 'complete row ' * 10000, 'tool_calls': []}
    surface = {'tools': [{'name': 'read', 'source': 'fallback' if fallback else 'predicted'}]}
    return {'schema': 'pw-training-row/1', 'session': session, 'message_id': 'm-' + session, 'source': {'bundle_sha256': bundle},
            'request_capture': capture, 'target': target, 'response_sha256': digest(target), 'prompt_surface': surface,
            'admission': {'eligibility': 'eligible', 'category': category, 'semantic_outcome': 'success'}}


def bank(tmp_path, monkeypatch, catalog=lambda task: 'k' * 64):
    journal = Journal(tmp_path / 'campaign', {})
    lineage = Lineage(journal, FOCUSES)
    specifications = [{'id': partition + family, 'partition': partition, 'focus': family, 'workspace': 'repo-' + family,
                       'source_class': 'synthetic', 'required_checks': ['outcome']}
                      for partition in sorted(PARTITIONS) for family in FOCUSES]
    register_bank(lineage, [dict(task) for task in specifications])
    lineage.freeze()
    rows, bundles = [], {}
    for index, task in enumerate(specifications):
        session = task['id']
        lineage.bind_session('epoch', session, task['id'])
        spec = {key: value for key, value in task.items() if key != 'partition'}
        receipt = {'task_spec_sha256': digest(spec), 'grader_identity': 'grader', 'complete': True,
                   'root_session': session, 'epoch': 'epoch', 'store_sha256': 'store',
                   'checks': [{'id': 'outcome', 'passed': True, 'source_sha256': 'final'}],
                   'final_source_sha256': 'final', 'evidence_ids': ['check'], 'human_interventions': 0,
                   'unauthorized_effects': 0, 'unresolved_effects': 0, 'teacher_family': 'teacher'}
        reviews = [{'reviewer_slot': name, 'judge_id': name, 'judge_family': name, 'episode_sha256': digest(receipt), 'controls_passed': True,
                    'evidence_ids': ['check'], 'dimensions': dict.fromkeys(DIMENSIONS, 'pass')} for name in ('first', 'second')]
        Episodes(journal, lineage).record('epoch', session, task['id'], [session], spec, receipt, reviews, 'grader')
        messages = COMPACTED if index % 4 == 0 else []
        bundle = write_episode(tmp_path / 'bundles' / session, sessions=[{'id': session, 'messages': messages}], store='store')
        bundles[str(bundle)] = file_sha256(bundle / 'manifest.json')
        rows.append(row(session, bundles[str(bundle)], 'recovery' if index % 5 == 0 else 'ordinary', index % 3 == 0, catalog(task)))
    manifests = {}
    for name, model in load()['models'].items():
        folder = tmp_path / name
        folder.mkdir()
        accepted, admissions = folder / 'rows.jsonl', folder / 'admissions.jsonl'
        accepted.write_text(''.join(json.dumps(item) + '\n' for item in rows))
        admissions.write_text(json.dumps({'admission': {'eligibility': 'excluded', 'reason': 'unknown_effects'}}) + '\n')
        manifest = folder / 'manifest.json'
        manifest.write_text(json.dumps({'schema': 'pw-training-rows/1', 'identity': {
            'host_commit': 'host', 'epoch': 'epoch', 'student_model': model['directory']},
            'bundles': bundles, 'rows_sha256': file_sha256(accepted), 'admissions_sha256': file_sha256(admissions)}))
        manifests[name] = [manifest]
    monkeypatch.setattr('bialy.campaign.dataset.training_ready', lambda row: True)
    strata = {task['id']: 'p0' for task in specifications}
    return journal, lineage, manifests, strata


def measure(limit=None):
    """Token counts as the host's encoder would give them, with `limit` sessions oversize."""
    def run(student, rows):
        out = []
        for line in rows.read_text().splitlines():
            session = json.loads(line)['session']
            out.append({'refused': 'sequence_over_limit', 'tokens': MAX_TOKENS + 1} if limit and session in limit
                       else {'tokens': 1000 + len(session)})
        return out
    return run


def sealer(seals):
    def seal(journal, lineage, paths, manifests, decisions, checkout, host):
        seals.append(paths)
        (paths[0].parent / 'source.json').write_text('sealed')
    return seal


def build(tmp_path, journal, lineage, manifests, strata, name='dataset', minimum=MINIMUM, measured=None, seals=None):
    return assemble(journal, lineage, manifests, {}, tmp_path, 'host', load(), tmp_path / name, minimum, P0, strata,
                    measured or measure(), seal=sealer([] if seals is None else seals))


def test_paired_datasets_use_registered_partitions_and_preserve_whole_rows(tmp_path, monkeypatch):
    journal, lineage, manifests, strata = bank(tmp_path, monkeypatch)
    seals = []
    result = build(tmp_path, journal, lineage, manifests, strata, seals=seals)
    assert len(seals) == len(students()) and result['data_status'] == 'data_ready'
    for name in students():
        assert result['students'][name]['episodes'] == MINIMUM
        rows = [json.loads(line) for line in (tmp_path / 'dataset' / name / 'train.jsonl').read_text().splitlines()]
        assert all(row['split_group'].startswith('train') for row in rows)
        assert all(len(row['target']['text']) == 130000 for row in rows)
    journal.close()


def test_the_data_report_is_reproducible_and_counts_what_the_rows_exercise(tmp_path, monkeypatch):
    journal, lineage, manifests, strata = bank(tmp_path, monkeypatch)
    first = build(tmp_path, journal, lineage, manifests, strata, 'one')
    second = build(tmp_path, journal, lineage, manifests, strata, 'two')
    assert (tmp_path / 'one' / 'data-report.json').read_bytes() == (tmp_path / 'two' / 'data-report.json').read_bytes()
    assert first['data_report_sha256'] == second['data_report_sha256']
    report = json.loads((tmp_path / 'one' / 'data-report.json').read_text())
    ledger = report['students']['e4b']
    assert ledger['independent']['train'] == {'groups': 8, 'repositories': 8, 'episodes': 8, 'encodings': 8}
    assert sum(sum(values.values()) for values in ledger['categories'].values()) == 24
    assert ledger['strata']['p0'] + ledger['strata']['runtime_fallback'] == 24 and ledger['strata']['runtime_fallback'] == 8
    assert ledger['exercise']['compacted_episodes'] == 6 and ledger['exclusions'] == {'development_partition': 8, 'unknown_effects': 1}
    assert ledger['tokens']['train']['max'] <= MAX_TOKENS and report['selector'] == {'policy': 'p0', 'fallback_rate': 0.1}
    text = (tmp_path / 'one' / 'data-report.md').read_text()
    assert 'complete row' not in text and 'Data status: data_ready' in text
    journal.close()


def test_an_oversize_sequence_is_refused_whole_and_reported_as_a_deficit(tmp_path, monkeypatch):
    journal, lineage, manifests, strata = bank(tmp_path, monkeypatch)
    seals = []
    with pytest.raises(CampaignFault, match='data_incomplete') as raised:
        build(tmp_path, journal, lineage, manifests, strata, measured=measure({'train' + FOCUSES[0]}), seals=seals)
    assert not seals and not (tmp_path / 'dataset' / 'e4b' / 'train.jsonl').exists()
    assert raised.value.details['deficits']['e4b'] == {'train': {'episodes_short': 1, 'focuses_missing': [FOCUSES[0]]}}
    report = json.loads((tmp_path / 'dataset' / 'data-report.json').read_text())
    assert report['data_status'] == 'data_incomplete' and report['students']['e4b']['exclusions']['sequence_over_limit'] == 1
    journal.close()


def test_missing_episode_coverage_never_publishes_training_permits(tmp_path, monkeypatch):
    journal, lineage, manifests, strata = bank(tmp_path, monkeypatch)
    seals = []
    with pytest.raises(CampaignFault, match='data_incomplete') as raised:
        build(tmp_path, journal, lineage, manifests, strata, minimum={'train': 32, 'validation': 16, 'system_test': 8}, seals=seals)
    assert not seals and raised.value.details['deficits']['e4b']['train'] == {'episodes_short': 24, 'focuses_missing': []}
    journal.close()


def test_rows_from_two_catalogs_are_refused(tmp_path, monkeypatch):
    journal, lineage, manifests, strata = bank(tmp_path, monkeypatch, catalog=lambda task: ('a' if task['focus'] == FOCUSES[0] else 'b') * 64)
    with pytest.raises(CampaignFault, match='dataset_epoch_mixed'):
        build(tmp_path, journal, lineage, manifests, strata)
    journal.close()


def test_a_row_whose_target_no_longer_matches_its_hash_is_refused(tmp_path, monkeypatch):
    journal, lineage, manifests, strata = bank(tmp_path, monkeypatch)
    for paths in manifests.values():
        manifest = paths[0]
        accepted = manifest.parent / 'rows.jsonl'
        rows = [json.loads(line) for line in accepted.read_text().splitlines()]
        next(item for item in rows if item['session'].startswith('train'))['target']['text'] = 'edited'
        accepted.write_text(''.join(json.dumps(item) + '\n' for item in rows))
        document = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({**document, 'rows_sha256': file_sha256(accepted)}))
    with pytest.raises(CampaignFault, match='row_hash_mismatch'):
        build(tmp_path, journal, lineage, manifests, strata)
    journal.close()


def test_runtime_fallback_is_its_own_stratum_only_under_p0():
    fallback = {'prompt_surface': {'tools': [{'source': 'fallback'}]}}
    assert coverage.stratum('p0', fallback) == 'runtime_fallback'
    assert coverage.stratum('planned_fallback', fallback) == 'planned_fallback'
    assert coverage.stratum('p0', {'prompt_surface': {'tools': [{'source': 'standing'}]}}) == 'p0'
    with pytest.raises(CampaignFault, match='row_stratum_unknown'):
        coverage.stratum(None, fallback)


def test_measurement_refuses_an_oversize_row_whole():
    class Oversize(ValueError):
        reason, tokens = 'oversize', MAX_TOKENS + 1
    class Modified(ValueError):
        reason = 'assistant_content_modified'
    def encode(item):
        if item == 'big':
            raise Oversize()
        if item == 'odd':
            raise Modified()
        if item == 'plain':
            raise ValueError('plain')
        return {'input_ids': [1, 2, 3]}
    assert measure_rows.measure(['ok', 'big', 'odd'], encode) == [
        {'tokens': 3}, {'refused': 'sequence_over_limit', 'tokens': MAX_TOKENS + 1},
        {'refused': 'assistant_content_modified', 'tokens': None}]
    with pytest.raises(ValueError, match='plain'):
        measure_rows.measure(['plain'], encode)
