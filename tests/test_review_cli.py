import json

import pytest

from bialy import cli
from bialy.campaign.human_review import CollectionGate, EvaluationGate, subject
from bialy.review import render


def bialy(campaign, *arguments):
    return cli.main(['review', '--run', str(campaign.journal.directory), *arguments])


def queued(campaign, kind='split', name='a'):
    task = campaign.task('train-' + name, partition='train')
    exports = {'e4b': {'manifest': 'm', 'manifest_sha256': '', 'rows': 7}}
    episode = {'stage': name, 'episode_sha256': 'episode-' + name, 'repetition': 0, 'exports': exports}
    reviews = ('pass', 'fail') if kind == 'split' else ('pass', 'pass')
    return CollectionGate(campaign.review, 'collection_pilot', 'command').offer_episode(episode, campaign.trial(name, reviews=reviews), task)


def pause(campaign, queue='collection_pilot'):
    (campaign.journal.directory / 'waiting.json').write_text(json.dumps({
        'reason': 'human_review', 'queue': queue, 'left': 1, 'command': 'bialy review', 'stage': 'review_' + queue,
        'resume': 'python -m bialy.campaign.pilot --configuration pilot.json'}))


def test_a_directory_without_a_campaign_is_reported(tmp_path, capsys):
    assert cli.main(['review', '--run', str(tmp_path)]) == 2
    assert 'No campaign at' in capsys.readouterr().err


def test_status_reports_queues_and_whether_the_campaign_waits(review_campaign, capsys):
    assert bialy(review_campaign) == 0
    assert 'No review queues are open yet.' in capsys.readouterr().out
    queued(review_campaign)
    pause(review_campaign)
    assert bialy(review_campaign, 'status') == 3
    out = capsys.readouterr().out
    assert 'Collection (pilot)' in out and '1 left · 0 labelled · 0 skipped' in out and 'waiting on you' in out
    assert bialy(review_campaign, 'status', '--json') == 3
    state = json.loads(capsys.readouterr().out)
    assert state['waiting_on_you'] == ['collection_pilot'] and state['queues'][0]['open'] == 1
    assert bialy(review_campaign, 'finish', 'collection_pilot', '--yes') == 0
    assert 'Finished Collection (pilot). 1 items were decided automatically.' in capsys.readouterr().out
    assert bialy(review_campaign, 'status') == 0
    assert 'python -m bialy.campaign.pilot --configuration pilot.json' in capsys.readouterr().out
    (review_campaign.journal.directory / 'waiting.json').unlink()
    assert bialy(review_campaign, 'status') == 0
    assert 'not waiting on you' in capsys.readouterr().out


def test_show_prints_the_transcript_with_short_evidence_references(review_campaign, capsys):
    item = queued(review_campaign)
    assert bialy(review_campaign, 'show', '#' + str(item)) == 0
    out = capsys.readouterr().out
    assert f'#{item} · Collection (pilot) · disagreement · build_test_repair · 7 rows' in out
    assert 'One reviewer failed this. Pass recovers 7 training rows' in out
    assert '→ command {"argv": ["go", "test", "./..."]} [r1]' in out and '[r1] completed' in out
    assert '… 8 more lines (show --full)' in out and 'Not decided.' in out
    assert bialy(review_campaign, 'show', str(item), '--full') == 0
    assert 'FAIL line 19' in capsys.readouterr().out


def test_skip_and_finish_are_final(review_campaign, capsys, monkeypatch):
    item = queued(review_campaign)
    assert bialy(review_campaign, 'skip', str(item)) == 0
    assert review_campaign.review.item(item)['state'] == 'skipped'
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda prompt: 'n')
    assert bialy(review_campaign, 'finish', 'collection_pilot') == 0
    assert 'Not finished.' in capsys.readouterr().out
    monkeypatch.setattr('builtins.input', lambda prompt: 'y')
    assert bialy(review_campaign, 'finish', 'collection_pilot') == 0
    assert bialy(review_campaign, 'skip', str(item)) == 2
    assert 'queue is finished; its decisions are final' in capsys.readouterr().err
    assert bialy(review_campaign, 'skip', 'nope') == 2
    assert bialy(review_campaign, 'show', '99') == 2
    assert bialy(review_campaign, 'finish', 'absent', '--yes') == 2
    assert 'No such queue.' in capsys.readouterr().err


def test_wait_returns_when_the_waiting_queue_is_finished(review_campaign):
    queued(review_campaign)
    assert bialy(review_campaign, 'wait') == 0
    pause(review_campaign)
    assert bialy(review_campaign, 'wait', '--timeout', '0', '--interval', '0') == 3
    review_campaign.review.finish('collection_pilot')
    assert bialy(review_campaign, 'wait', '--interval', '0') == 0



def test_wait_polls_until_the_reviewer_finishes(review_campaign, monkeypatch):
    queued(review_campaign)
    pause(review_campaign)
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        review_campaign.review.finish('collection_pilot')
    monkeypatch.setattr('bialy.review.cli.time.sleep', sleep)
    assert bialy(review_campaign, 'wait', '--interval', '5') == 0
    assert sleeps == [5.0]


def test_rendered_decisions_and_instructions_say_what_each_item_decides(review_campaign):
    spot = review_campaign.review.item(queued(review_campaign, 'spot_check', 'b'))
    assert render.instruction(spot) == 'Both reviewers passed this. Fail removes 7 rows; pass or skip keeps them.'
    review_campaign.task('held-out')
    evaluation = {'kind': 'pair', 'queue': 'evaluation', 'summary': {'family': 'build_test_repair', 'student': 'e4b'}}
    assert render.instruction(evaluation) == "Grade each side on its own. Don't pick a winner."
    assert render.instruction({**evaluation, 'kind': 'split'}) == "Grade this run. Your verdict replaces the reviewers'."
    assert render.heading({**evaluation, 'id': 4, 'summary': {**evaluation['summary'], 'arm': 'base'}}) == \
        '#4 · Evaluation · pair · build_test_repair · e4b · base'
    item = review_campaign.review.item(queued(review_campaign))
    view = render.Subject(item['subjects'][0])
    dimensions, citations = review_campaign.verdict()
    decision = {'decided': 'human', 'dimensions': {**dimensions, 'verification': 'fail'},
                'citations': {**citations, 'verification': ['invocation-1']}, 'reason': 'one test only', 'revealed_reviewers': True}
    lines = render.decision_lines(item, view, decision)
    assert lines[0] == 'Labelled fail after revealing reviewers' and 'cite r1' in lines[4] and lines[-1] == '  Reason  one test only'
    assert render.decision_lines(item, view, {'decided': 'automatic'}) == ['Skipped: decided automatically.']
    assert view.reviewer_lines() == ['  judge-0: pass', '    reviewer 0 reason',
                                     '  judge-1: fail on ' + ', '.join(title for title, _ in render.CHECKS.values()),
                                     '    reviewer 1 reason']
    assert render.queue_title('custom') == 'custom' and set(render.CHECKS) == set(dimensions)


def test_pairs_render_both_sides(review_campaign):
    review_campaign.task('t1')
    trials = {arm: review_campaign.trial('x-' + arm, reviews=('pass', 'fail') if arm == 'base' else ('pass', 'pass'))
              for arm in ('base', 'trained')}
    gate = EvaluationGate(review_campaign.review, 'evaluation', review_campaign.journal, 'command')
    review_campaign.review.open('evaluation', 'evaluation')
    item = review_campaign.review.offer('evaluation', 'pair', [subject('x-' + arm, trials[arm], arm) for arm in trials],
                                        [{'side': 'base', 'trial': 'x-base', 'success': False},
                                         {'side': 'trained', 'trial': 'x-trained', 'success': True}],
                                        {**gate.summary({'task_id': 't1', 'student': 'e4b', 'attempt': 0})}, 0)
    text = render.item_text(review_campaign.review.item(item))
    assert '  base: failure' in text and '  trained: success' in text and '\nBase\n' in text and '\nTrained\n' in text


def test_workers_appear_in_the_transcript(review_campaign):
    item = review_campaign.review.item(queued(review_campaign))
    path = review_campaign.journal.directory / 'trials' / 'a' / 'host-evidence.json'
    facts = json.loads(path.read_text())
    facts['sessions'].append({'id': 'worker', 'parent_id': 'root', 'messages': [{'role': 'assistant', 'content': ''}]})
    path.write_text(json.dumps(facts))
    assert '  ── worker worker' in render.item_text(item)


@pytest.mark.parametrize('text', ['', 'one\n\ntwo'])
def test_wrapping_keeps_blank_lines(text):
    assert render.wrap(text, 40) == (text.splitlines() or [''])
