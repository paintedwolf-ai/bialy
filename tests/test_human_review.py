import json

import pytest

from bialy.campaign.dataset import Assembly
from bialy.campaign.evaluation import Comparison
from bialy.campaign.execution import Executor
from bialy.campaign.grading import DIMENSIONS, effective_attempts, reviewed
from bialy.campaign.human_review import CollectionGate, EvaluationGate
from bialy.campaign.journal import CampaignFault, Journal, StageWaiting


def subject(name):
    return {'episode_sha256': 'episode-' + name, 'trial': name, 'evidence_ids': ['invocation-1', 'tests'], 'side': None}


def test_trials_classify_into_pass_split_and_fail(review_campaign):
    assert reviewed(review_campaign.trial('a')) == 'pass'
    assert reviewed(review_campaign.trial('b', reviews=('pass', 'fail'))) == 'split'
    assert reviewed(review_campaign.trial('c', reviews=('fail', 'fail'))) == 'fail'
    assert reviewed(review_campaign.trial('d', deterministic=False)) == 'fail'
    assert reviewed(review_campaign.trial('e', exit_status=1)) == 'fail'


def test_each_episode_belongs_to_one_item_and_labels_are_validated(review_campaign):
    review = review_campaign.review
    review.open('collection_pilot', 'collection')
    item = review.offer('collection_pilot', 'split', [subject('a')], [], {'rows': 4}, 4)
    assert review.offer('collection_pilot', 'spot_check', [subject('a')], [], {'rows': 4}, 4) is None
    dimensions, citations = review_campaign.verdict()
    for bad in ({**dimensions, 'task_outcome': 'maybe'}, {key: dimensions[key] for key in DIMENSIONS[1:]}):
        with pytest.raises(CampaignFault, match='dimensions_invalid'):
            review.label(item, 'episode-a', bad, citations, '')
    for bad in ({**citations, 'verification': []}, {**citations, 'verification': ['invented']}):
        with pytest.raises(CampaignFault, match='citations_invalid'):
            review.label(item, 'episode-a', dimensions, bad, '')
    with pytest.raises(CampaignFault, match='subject_missing'):
        review.label(item, 'episode-b', dimensions, citations, '')
    with pytest.raises(CampaignFault, match='decision_invalid'):
        review.label(item, 'episode-a', dimensions, citations, None)
    assert review.item(item)['state'] == 'open'
    review.label(item, 'episode-a', dimensions, citations, ' ran the tests ')
    assert review.item(item)['state'] == 'labelled'
    assert review.item(item)['decisions']['episode-a']['revealed_reviewers'] is False
    review.reveal(item, 'episode-a')
    review.reveal(item, 'episode-a')
    with pytest.raises(CampaignFault, match='subject_missing'):
        review.reveal(item, 'episode-b')
    review.label(item, 'episode-a', {**dimensions, 'verification': 'fail'}, citations, '')
    latest = review.item(item)['decisions']['episode-a']
    assert latest['dimensions']['verification'] == 'fail' and latest['schema'] == 'pw-human-review-decision/1'
    assert latest['revealed_reviewers'] is True and latest['reason'] == ''
    review.skip(item)
    assert review.item(item)['state'] == 'skipped'
    with pytest.raises(CampaignFault, match='kind_invalid'):
        review.offer('collection_pilot', 'other', [subject('z')], [], {}, 0)
    with pytest.raises(CampaignFault, match='queue_missing'):
        review.offer('absent', 'split', [subject('z')], [], {}, 0)
    with pytest.raises(CampaignFault, match='item_missing'):
        review.item(999)


def test_finished_queues_are_final_and_freeze_only_human_decisions(review_campaign):
    review = review_campaign.review
    review.open('collection_pilot', 'collection')
    labelled = review.offer('collection_pilot', 'split', [subject('a')], [], {'rows': 2}, 2)
    skipped = review.offer('collection_pilot', 'spot_check', [subject('b')], [], {'rows': 9}, 9)
    left = review.offer('collection_pilot', 'spot_check', [subject('c')], [], {'rows': 1}, 1)
    assert [item['id'] for item in review.items('collection_pilot')] == [labelled, skipped, left]
    review.label(labelled, 'episode-a', *review_campaign.verdict(), 'recovered')
    review.skip(skipped)
    with pytest.raises(CampaignFault, match='unfinished'):
        review.frozen('collection_pilot', ['episode-a'])
    review.finish('collection_pilot')
    review.finish('collection_pilot')
    assert review.counts('collection_pilot') == {'open': 0, 'labelled': 1, 'skipped': 1, 'automatic': 1}
    with pytest.raises(CampaignFault, match='finished'):
        review.skip(left)
    frozen = review.frozen('collection_pilot', ['episode-a', 'episode-b', 'episode-c', 'episode-unknown'])
    assert set(frozen['decisions']) == {'episode-a'} and frozen['decisions']['episode-a']['reason'] == 'recovered'
    with pytest.raises(CampaignFault, match='queue_missing'):
        review.finish('absent')
    with pytest.raises(CampaignFault, match='kind_changed'):
        review.open('collection_pilot', 'evaluation')
    with pytest.raises(CampaignFault, match='kind_invalid'):
        review.open('other', 'other')


def test_later_queues_reuse_decisions_from_earlier_finished_queues(review_campaign):
    review = review_campaign.review
    review.open('collection_pilot', 'collection')
    item = review.offer('collection_pilot', 'split', [subject('a')], [], {'rows': 2}, 2)
    review.label(item, 'episode-a', *review_campaign.verdict(), '')
    review.open('collection_full', 'collection')
    review.offer('collection_full', 'split', [subject('b')], [], {'rows': 2}, 2)
    with pytest.raises(CampaignFault, match='unfinished'):
        review.frozen('collection_full', ['episode-a'])
    review.finish('collection_full')
    with pytest.raises(CampaignFault, match='unfinished'):
        review.frozen('collection_full', ['episode-a'])
    review.finish('collection_pilot')
    assert set(review.frozen('collection_full', ['episode-a', 'episode-b'])['decisions']) == {'episode-a'}


def test_collection_gate_waits_then_passes_frozen_decisions(review_campaign):
    review = review_campaign.review
    gate = CollectionGate(review, 'collection_pilot', 'bialy review --run campaign')
    task = review_campaign.task('train-a', partition='train')
    exports = {'e4b': {'manifest': 'm', 'manifest_sha256': '', 'rows': 3}, '12b': {'manifest': 'm', 'manifest_sha256': '', 'rows': 5}}
    episode = {'stage': 'a', 'episode_sha256': 'episode-a', 'repetition': 0, 'exports': exports}
    item = gate.offer_episode(episode, review_campaign.trial('a', reviews=('pass', 'fail')), task)
    assert review.item(item)['kind'] == 'split' and review.item(item)['summary']['rows'] == 3
    assert gate.offer_episode({**episode, 'stage': 'b', 'episode_sha256': 'episode-b'},
                              review_campaign.trial('b', reviews=('fail', 'fail')), task) is None
    assert gate.offer_episode({**episode, 'stage': 'c', 'episode_sha256': 'episode-c', 'exports': None},
                              review_campaign.trial('c'), task) is None
    inputs = {'collection': {'episodes': [episode]}}
    with pytest.raises(StageWaiting) as waiting:
        gate.execute(inputs, 'attempt')
    assert waiting.value.record() == {'reason': 'human_review', 'queue': 'collection_pilot', 'left': 1,
                                      'command': 'bialy review --run campaign'}
    review.label(item, 'episode-a', *review_campaign.verdict(), '')
    with pytest.raises(StageWaiting):
        gate.execute(inputs, 'attempt')
    review.finish('collection_pilot')
    output = gate.execute(inputs, 'attempt')
    gate.validate(inputs, output)
    assert set(output['decisions']) == {'episode-a'}
    with pytest.raises(CampaignFault, match='decisions_changed'):
        gate.validate(inputs, {**output, 'decisions': {}})
    assert gate.reconcile(inputs, 'attempt') == output


def test_an_empty_queue_never_waits(review_campaign):
    gate = CollectionGate(review_campaign.review, 'collection_pilot', 'command')
    output = gate.execute({'collection': {'episodes': []}}, 'attempt')
    assert output['decisions'] == {} and review_campaign.review.queue('collection_pilot')['finished_at']


def attempt(campaign, student, arm, task, success, review, name):
    campaign.trial(name, reviews=('pass', 'pass') if success else ('pass', 'fail'), deterministic=review != 'fail')
    return {'task_id': task, 'attempt': 0, 'arm': arm, 'student': student, 'trial': name, 'success': success,
            'review': review, 'episode_sha256': 'episode-' + name if review else None, 'stage_status': 'passed',
            'human_interventions': 0, 'unauthorized_effects': 0, 'unsupported_completion_claims': 0, 'effective_thinking': False}


def test_evaluation_gate_offers_differing_pairs_then_remaining_disagreements(review_campaign):
    for task in ('t1', 't2', 't3', 't4'):
        review_campaign.task(task)
    evaluations = {'e4b': {
        'base': {'attempts': [attempt(review_campaign, 'e4b', 'base', 't1', False, 'split', 'b1'),
                              attempt(review_campaign, 'e4b', 'base', 't2', True, 'pass', 'b2'),
                              attempt(review_campaign, 'e4b', 'base', 't3', False, 'fail', 'b3'),
                              attempt(review_campaign, 'e4b', 'base', 't4', False, 'fail', 'b4')]},
        'trained': {'attempts': [attempt(review_campaign, 'e4b', 'trained', 't1', True, 'pass', 'r1'),
                                 attempt(review_campaign, 'e4b', 'trained', 't2', False, 'split', 'r2'),
                                 attempt(review_campaign, 'e4b', 'trained', 't3', False, 'split', 'r3'),
                                 attempt(review_campaign, 'e4b', 'trained', 't4', False, 'fail', 'r4')]}}}
    gate = EvaluationGate(review_campaign.review, 'evaluation', review_campaign.journal, 'command')
    with pytest.raises(StageWaiting):
        gate.execute({'evaluations': evaluations}, 'attempt')
    items = review_campaign.review.items('evaluation')
    assert [(item['kind'], [subject['side'] for subject in item['subjects']]) for item in items] == [
        ('split', ['trained']), ('pair', ['base', 'trained']), ('pair', ['base', 'trained'])]
    assert items[1]['summary']['arms'] == {'base': False, 'trained': True}
    assert items[0]['summary']['prompt'].startswith('Add a --json flag')


def test_human_decisions_change_evaluation_grades_but_not_failed_checks(review_campaign):
    dimensions, citations = review_campaign.verdict()
    attempts = [{'episode_sha256': 'x', 'review': 'split', 'success': False, 'unsupported_completion_claims': 1},
                {'episode_sha256': 'y', 'review': 'pass', 'success': True, 'unsupported_completion_claims': 0},
                {'episode_sha256': None, 'review': 'fail', 'success': False, 'unsupported_completion_claims': None}]
    decisions = {'x': {'dimensions': dimensions}, 'y': {'dimensions': {**dimensions, 'truthful_closeout': 'fail'}}}
    first, second, third = effective_attempts(attempts, decisions)
    assert first['success'] is True and first['unsupported_completion_claims'] == 0
    assert second['success'] is False and second['unsupported_completion_claims'] == 1
    assert third is attempts[2]
    with pytest.raises(CampaignFault, match='failed_checks'):
        effective_attempts([{**attempts[2], 'episode_sha256': 'x'}], decisions)


def test_comparison_grades_with_frozen_evaluation_decisions(monkeypatch):
    seen = []
    monkeypatch.setattr('bialy.campaign.evaluation.behavioral_result', lambda plan, attempts, student: seen.append(attempts) or {})
    attempt = {'episode_sha256': 'x', 'review': 'split', 'success': False, 'unsupported_completion_claims': 1}
    decision = {'dimensions': dict.fromkeys(DIMENSIONS, 'pass')}
    inputs = {'plan': {}, 'student': 'e4b', 'base': {'attempts': [attempt]}, 'trained': {'attempts': []}}
    Comparison().execute({**inputs, 'review': {'decisions': {'x': decision}}}, None)
    Comparison().execute({**inputs, 'review': None}, None)
    assert seen[0][0]['success'] is True and seen[1][0]['success'] is False


def collected(task, repetition, admitted, rows=3):
    exports = {'e4b': {'manifest': f'{task}-{repetition}', 'manifest_sha256': '', 'rows': rows}}
    return {'stage': f'{task}-{repetition}', 'task_id': task, 'repetition': repetition, 'admitted': admitted,
            'episode_sha256': f'{task}-{repetition}', 'exports': exports}


def test_assembly_applies_human_decisions_before_choosing_each_task(tmp_path):
    journal = Journal(tmp_path, {})
    seen = {}
    def assemble(manifests, decisions, directory, minimum):
        seen.update(manifests=manifests, decisions=decisions)
        return {'students': {'e4b': {'directory': str(directory), 'files': {}}}}
    episodes = [collected('a', 0, False), collected('a', 1, True), collected('b', 0, True), collected('c', 0, False, rows=0),
                {**collected('d', 0, False), 'exports': None}]
    passed, failed = dict.fromkeys(DIMENSIONS, 'pass'), {**dict.fromkeys(DIMENSIONS, 'pass'), 'verification': 'fail'}
    review = {'decisions': {'a-0': {'dimensions': passed}, 'b-0': {'dimensions': failed},
                            'c-0': {'dimensions': passed}, 'd-0': {'dimensions': passed}}}
    stage = Assembly(journal, ['e4b'], assemble)
    inputs = {'collection': {'episodes': episodes, 'minimum': {}}, 'review': review}
    output = stage.execute(inputs, 'attempt')
    stage.validate(inputs, output)
    assert output['episodes'] == ['a-0'] and seen['manifests'] == {'e4b': ['a-0']}
    assert set(seen['decisions']) == {'a-0', 'b-0', 'c-0', 'd-0'}
    journal.close()


def test_a_waiting_stage_resumes_without_counting_a_retry(tmp_path):
    journal = Journal(tmp_path, {})
    calls = []
    class Gate:
        def execute(self, name, inputs, attempt):
            calls.append(attempt)
            if len(calls) == 1:
                raise StageWaiting('human_review', {'queue': 'q', 'left': 2, 'command': 'c'})
            return {'decisions': {}}
        def validate(self, name, inputs, output):
            pass
    executor = Executor(journal, Gate(), lambda _: None)
    first = executor.run('review_collection_pilot', {})
    assert first['status'] == 'waiting' and first['output'] == {'reason': 'human_review', 'queue': 'q', 'left': 2, 'command': 'c'}
    second = executor.run('review_collection_pilot', {})
    assert second['status'] == 'passed' and second['tries'] == 1 and len(set(calls)) == 2
    with pytest.raises(ValueError, match='settlement'):
        journal.settle('review_collection_pilot', second['attempt'], 'paused')
    journal.close()


def test_an_existing_campaign_opens_without_restating_its_identity(tmp_path):
    with pytest.raises(CampaignFault, match='campaign_missing'):
        Journal(tmp_path / 'absent', None)
    Journal(tmp_path, {'schema': 'fixture'}).close()
    reopened = Journal(tmp_path, None)
    assert reopened.manifest == {'schema': 'fixture'}
    reopened.close()
    empty = tmp_path / 'empty'
    empty.mkdir()
    import sqlite3
    sqlite3.connect(empty / 'campaign.sqlite').close()
    with pytest.raises(CampaignFault, match='campaign_missing'):
        Journal(empty, None)
    assert json.loads(json.dumps({'ok': True}))


def recorded_episode(journal, passed_checks, reviewer_values):
    from bialy.campaign.episodes import Episodes
    from bialy.campaign.journal import digest
    from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage
    lineage = Lineage(journal)
    tasks = [{'id': partition + family, 'partition': partition, 'family': family, 'source_class': 'synthetic',
              'required_checks': ['outcome']} for partition in sorted(PARTITIONS) for family in FAMILIES]
    lineage.roots(tasks)
    lineage.freeze()
    task = tasks[0]
    lineage.bind_session('epoch', 'session', task['id'])
    receipt = {'task_spec_sha256': digest(task), 'grader_identity': 'grader', 'complete': True, 'root_session': 'session',
               'epoch': 'epoch', 'store_sha256': 'store', 'final_source_sha256': 'final', 'evidence_ids': ['check'],
               'checks': [{'id': 'outcome', 'passed': passed_checks, 'source_sha256': 'final'}],
               'human_interventions': 0, 'unauthorized_effects': 0, 'unresolved_effects': 0}
    reviews = [{'reviewer_slot': str(index), 'episode_sha256': digest(receipt), 'controls_passed': True,
                'evidence_ids': ['check'], 'dimensions': dict.fromkeys(DIMENSIONS, value)}
               for index, value in enumerate(reviewer_values)]
    episodes = Episodes(journal, lineage)
    verdict = episodes.record('epoch', 'session', task['id'], ['session'], task, receipt, reviews, 'grader')
    return episodes, verdict


@pytest.mark.parametrize('passed_checks, decision, admitted', [
    (True, None, False), (True, 'pass', True), (True, 'fail', False), (False, 'pass', False)])
def test_admission_uses_a_human_decision_only_when_the_checks_passed(tmp_path, passed_checks, decision, admitted):
    from bialy.campaign.journal import digest
    journal = Journal(tmp_path, {})
    episodes, verdict = recorded_episode(journal, passed_checks, ('pass', 'fail'))
    assert verdict['schema'] == 'pw-episode-verdict/2' and verdict['deterministic'] is passed_checks
    assert verdict['semantic'] is False
    decisions = {verdict['episode_sha256']: {'dimensions': dict.fromkeys(DIMENSIONS, decision)}} if decision else {}
    if admitted:
        admission = episodes.admit('epoch', 'session', 'store', decisions)
        assert admission['decision_sha256'] == digest(decisions[verdict['episode_sha256']])
    else:
        with pytest.raises(CampaignFault, match='episode_not_qualified'):
            episodes.admit('epoch', 'session', 'store', decisions)
    journal.close()


def test_collection_exports_and_offers_episodes_a_label_could_change(review_campaign, tmp_path):
    from bialy.campaign.collection import Collection
    from bialy.campaign.lineage import FAMILIES, PARTITIONS
    tasks = {}
    for partition in sorted(PARTITIONS):
        for family in FAMILIES:
            tasks[partition + family] = review_campaign.task(partition + family, family=family, partition=partition)
    review_campaign.lineage.freeze()
    outcomes = {('pass', 'fail'): 'split', ('fail', 'fail'): 'fail', ('pass', 'pass'): 'pass'}
    plan = [('pass', 'fail'), ('fail', 'fail'), ('pass', 'pass')]
    class Driver:
        def execute(self, name, inputs, attempt):
            reviews = plan[inputs['repetition']]
            return review_campaign.trial(name, reviews=reviews, record=False)
        def validate(self, name, inputs, output):
            pass
    exported = []
    def export(inputs, output):
        exported.append(output['verdict']['episode_sha256'])
        manifest = tmp_path / (output['verdict']['episode_sha256'] + '.json')
        manifest.write_text('{}')
        return {'e4b': {'manifest': str(manifest), 'rows': 2}}
    gate = CollectionGate(review_campaign.review, 'collection_pilot', 'command')
    stage = Collection(review_campaign.journal, review_campaign.lineage, Driver(), export, ['e4b'], True,
                       minimum={'train': 8, 'validation': 8, 'system_test': 8}, review=gate)
    inputs = {'host': {}, 'selector': {}, 'teachers': {'teachers': [{'id': 'teacher'}],
              'families': {family: {'teacher_id': 'teacher'} for family in FAMILIES}}}
    output = stage.execute(inputs, 'attempt')
    kinds = [outcomes[plan[episode['repetition']]] for episode in output['episodes']]
    assert kinds == ['split', 'fail', 'pass'] * 24
    assert [episode['admitted'] for episode in output['episodes']] == [False, False, True] * 24
    assert all((episode['exports'] is None) == (kind == 'fail') for episode, kind in zip(output["episodes"], kinds, strict=True))
    items = review_campaign.review.items('collection_pilot')
    assert [item['kind'] for item in items].count('split') == 24 and [item['kind'] for item in items].count('spot_check') == 24
    assert len(exported) == 48 and all(item['summary']['rows'] == 2 for item in items)


def test_concurrent_decisions_take_the_next_receipt_number(review_campaign, monkeypatch):
    from bialy.campaign import human_review
    review = review_campaign.review
    review.open('collection_pilot', 'collection')
    item = review.offer('collection_pilot', 'split', [subject('a')], [], {'rows': 1}, 1)
    original, collisions = human_review.write, []
    def colliding(path, value, exclusive=False):
        if exclusive and path.parent.name == str(item) and not collisions:
            collisions.append(path.name)
            original(path, {**value, 'reason': 'another reviewer'}, exclusive=True)
            raise FileExistsError(path)
        return original(path, value, exclusive)
    monkeypatch.setattr(human_review, 'write', colliding)
    review.skip(item)
    receipts = review.receipts(review.item(item))
    assert collisions == ['1.json'] and [path.name for path in receipts] == ['1.json', '2.json']
    assert review.item(item)['decisions']['episode-a']['decided'] == 'automatic'


def test_evaluation_gate_freezes_decisions_for_its_attempts(review_campaign):
    review_campaign.task('t1')
    evaluations = {'e4b': {'base': {'attempts': [attempt(review_campaign, 'e4b', 'base', 't1', False, 'split', 'b1')]},
                           'trained': {'attempts': [attempt(review_campaign, 'e4b', 'trained', 't1', False, 'fail', 'r1')]}}}
    evaluations['e4b']['trained']['attempts'][0]['episode_sha256'] = None
    gate = EvaluationGate(review_campaign.review, 'evaluation', review_campaign.journal, 'command')
    with pytest.raises(StageWaiting):
        gate.execute({'evaluations': evaluations}, 'attempt')
    item = review_campaign.review.items('evaluation')[0]
    review_campaign.review.label(item['id'], 'episode-b1', *review_campaign.verdict(), '')
    review_campaign.review.finish('evaluation')
    output = gate.execute({'evaluations': evaluations}, 'attempt')
    assert set(output['decisions']) == {'episode-b1'}
    gate.validate({'evaluations': evaluations}, output)
