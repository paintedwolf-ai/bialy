import json

import pytest

from bialy import cli
from bialy.campaign.human_review import CollectionGate, subject
from bialy.review import terminal
from bialy.review.form import automatic, order, outcome
from bialy.review.model import Reviewer


def queued(campaign, name, kind='split', rows=7):
    task = campaign.task('train-' + name, partition='train')
    exports = {'e4b': {'manifest': 'm', 'manifest_sha256': '', 'rows': rows}}
    episode = {'stage': name, 'episode_sha256': 'episode-' + name, 'repetition': 0, 'exports': exports}
    reviews = ('pass', 'fail') if kind == 'split' else ('pass', 'pass')
    return CollectionGate(campaign.review, 'collection_pilot', 'command').offer_episode(episode, campaign.trial(name, reviews=reviews), task)


def reviewer(campaign, copies=None):
    def copy(text, command):
        if copies is None:
            raise OSError('no clipboard')
        copies.append((text, command))
        return 'Copied.'
    return Reviewer(campaign.review, campaign.journal.directory, copy)


def press(model, *keys):
    for key in keys:
        model.handle(key)
    return model


def screen(model, height=40, width=120):
    return '\n'.join(text for text, _ in model.render(height, width))


def test_the_welcome_appears_once_per_campaign(review_campaign):
    model = reviewer(review_campaign)
    assert 'Welcome to review' in screen(model) and 'A claim with no receipt fails.' in screen(model)
    press(model, '?', 'enter')
    assert model.screen == 'queues' and 'No review queues are open yet' in screen(model)
    assert reviewer(review_campaign).screen == 'queues'
    assert press(reviewer(review_campaign), 'q').done
    first = reviewer(review_campaign)
    (review_campaign.review.root / '.welcomed').unlink()
    assert press(reviewer(review_campaign), 'q').done and first.screen == 'queues'


def test_queues_and_items_are_listed_in_review_order(review_campaign):
    queued(review_campaign, 'spot', 'spot_check', rows=30)
    split = queued(review_campaign, 'split', rows=3)
    (review_campaign.journal.directory / 'waiting.json').write_text(json.dumps({'queue': 'collection_pilot', 'resume': 'resume-command'}))
    model = press(reviewer(review_campaign), 'enter')
    text = screen(model)
    assert 'Review' in text and 'Pick a queue.' in text
    assert '▸ Collection (pilot)' in text and '2 left · 0 labelled · 0 skipped   The campaign is waiting on you' in text
    press(model, 'up', 'k', 'down', 'j', 'enter')
    text = screen(model)
    assert 'Review › Collection (pilot)' in text and 'Should the students learn from these sessions?' in text
    rows = [line for line in text.splitlines() if 'build_test_repair' in line]
    assert rows[0].startswith(f'▸ {split}') and 'disagreement' in rows[0] and 'spot-check' in rows[1]
    press(model, '/', 'x', 'y', 'z', 'enter')
    assert 'No items match.' in screen(model)
    press(model, '/', 'esc')
    assert len(model.items) == 2
    press(model, '/', 's', 'p', 'o', 't', 'x', 'backspace', 'enter')
    assert len(model.items) == 1 and model.items[0]['kind'] == 'spot_check'
    press(model, 'esc', 'esc')
    assert model.screen == 'queues'


def test_an_item_shows_its_instruction_transcript_and_navigation(review_campaign):
    first = queued(review_campaign, 'a')
    second = queued(review_campaign, 'b')
    copies = []
    model = press(reviewer(review_campaign, copies), 'enter', 'enter', 'enter')
    text = screen(model, height=60)
    assert f'Review › Collection (pilot) › #{first} · disagreement · build_test_repair · 7 rows' in text
    assert 'One reviewer failed this. Pass recovers 7 training rows' in text and '[r1] completed' in text
    assert 'v verdict   x skip' in text and 'Tab other side' not in text
    screen(model, height=12)
    press(model, 'G')
    assert model.scroll > 0
    press(model, 'g', 'down', 'up', ' ', 'b', 'pgdn', 'pgup', 'home', 'end')
    press(model, '/', 'F', 'A', 'I', 'L', 'enter')
    assert model.matches and 'matches · n/N' in screen(model, height=60)
    press(model, 'n', 'N', '/', 'z', 'z', 'z', 'enter')
    assert 'No matches.' in screen(model)
    press(model, '|', 'enter')
    assert copies[0][1] == '' and copies[0][0].startswith(f'#{first}') and 'Copied.' in screen(model)
    press(model, '|', 'c', 'a', 't', 'enter')
    assert copies[1][1] == 'cat'
    press(model, '|', 'esc', 'right')
    assert model.item['id'] == second
    press(model, 'right', 'left', 'h')
    assert model.item['id'] == first
    press(model, '?')
    assert 'Help' in screen(model) and 'Was the asked-for work actually done?' in screen(model)
    press(model, 'x', 'esc')
    assert model.screen == 'items'
    press(model, 'enter', 'q')
    assert model.done


def test_copy_failures_are_reported(review_campaign):
    queued(review_campaign, 'a')
    model = press(reviewer(review_campaign), 'enter', 'enter', 'enter', '|', 'enter')
    assert 'Copy failed: no clipboard' in screen(model)


def test_a_verdict_needs_every_check_and_a_citation_then_saves(review_campaign):
    item = queued(review_campaign, 'a')
    model = press(reviewer(review_campaign), 'enter', 'enter', 'enter', 'r')
    assert 'Model reviewers' in screen(model, height=80) and 'judge-1: fail on' in screen(model, height=80)
    press(model, 'v')
    text = screen(model)
    assert 'Review › Collection (pilot) › #%d · disagreement' % item in text and '› Verdict' in text
    assert 'Set each check, cite a receipt, then press Enter.' in text and 'Result: set every check' in text
    press(model, 'enter')
    assert 'Set this check to pass or fail.' in screen(model)
    press(model, '1', 'p', 'p', 'p', 'p', 'p', 'f', 'enter')
    assert 'Cite a receipt for this check.' in screen(model) and 'Result: fail → stays out of training' in screen(model)
    for check in '123456':
        press(model, check, 'c')
        if check == '1':
            assert 'Pick the receipts that show this check. Space toggles.' in screen(model)
            assert '[ ] tests            check passed' in screen(model) and 'r1               command completed' in screen(model)
            press(model, 'down', 'up', ' ', 'esc', 'c')
        press(model, 'down', ' ', 'k', 'j', 'enter')
    press(model, '6', 'p', 'e', 'o', 'k', 'x', 'backspace', 'enter')
    text = screen(model)
    assert 'cite r1' in text and 'Reason (optional)  ok' in text and 'Result: pass → admits 7 training rows' in text
    press(model, 'enter')
    decision = review_campaign.review.item(item)['decisions']['episode-a']
    assert decision['revealed_reviewers'] is True and decision['reason'] == 'ok'
    assert decision['citations']['task_outcome'] == ['invocation-1']
    assert 'Saved #%d: pass → admits 7 training rows.' % item in screen(model)
    press(model, 'u', 'esc', 'esc')
    assert 'passed' in [line for line in screen(model).splitlines() if 'build_test_repair' in line][0]


def test_skip_and_finish_confirm_then_are_final(review_campaign):
    item = queued(review_campaign, 'a', 'spot_check')
    model = press(reviewer(review_campaign), 'enter', 'enter', 'enter', 'x')
    assert f'Skip #{item}? It will be decided automatically: stays in training.' in screen(model)
    press(model, 'n')
    assert review_campaign.review.item(item)['state'] == 'open'
    press(model, 'x', 'y')
    assert review_campaign.review.item(item)['state'] == 'skipped' and 'Skipped' in screen(model)
    press(model, 'esc', 'F')
    assert 'Finish the Collection (pilot) queue? 0 items left will be decided automatically.' in screen(model)
    press(model, 'y')
    assert 'The campaign was not waiting on this queue.' in screen(model)
    press(model, 'F')
    assert 'already finished' in screen(model)
    press(model, 'enter', 'v')
    assert model.form is None and 'finished; its decisions are final' in screen(model)
    press(model, 'x')
    assert 'finished; its decisions are final' in screen(model)


def test_finishing_a_waited_queue_shows_how_to_resume(review_campaign):
    queued(review_campaign, 'a')
    (review_campaign.journal.directory / 'waiting.json').write_text(json.dumps({'queue': 'collection_pilot', 'resume': 'resume-command'}))
    model = press(reviewer(review_campaign), 'enter', 'enter', 'F', 'y')
    assert 'Finished. Resume the campaign with: resume-command' in screen(model)
    assert 'Finished' in screen(press(model, 'esc'))


def test_pairs_switch_sides_with_tab(review_campaign):
    review_campaign.task('t1')
    trials = {arm: review_campaign.trial('x-' + arm, reviews=('pass', 'fail') if arm == 'base' else ('pass', 'pass'))
              for arm in ('base', 'trained')}
    review_campaign.review.open('evaluation', 'evaluation')
    review_campaign.review.offer('evaluation', 'pair', [subject('x-' + arm, trials[arm], arm) for arm in trials],
                                 [{'side': 'base', 'trial': 'x-base', 'success': False},
                                  {'side': 'trained', 'trial': 'x-trained', 'success': True}],
                                 {'task_id': 't1', 'family': 'build_test_repair', 'student': 'e4b', 'repetition': 0,
                                  'prompt': 'Add a flag', 'arms': {'base': False, 'trained': True}}, 0)
    model = press(reviewer(review_campaign), 'enter', 'enter', 'enter')
    text = screen(model)
    assert "Did each student's run actually succeed?" not in text and "Grade each side on its own." in text
    assert 'Showing base · Tab for the other side' in text and 'Tab other side' in text
    press(model, 'tab')
    assert 'Showing trained' in screen(model)
    press(model, 'v', '1', 'p')
    assert 'Result: set every check' in screen(model)
    press(model, *'pppppp')
    assert 'Result: pass → counts as a success' in screen(model)
    press(model, 'esc', 'esc')
    assert 'labelled' not in screen(model)


def test_plain_words_for_outcomes():
    item = {'queue': 'evaluation', 'kind': 'split', 'summary': {}}
    assert outcome(item, False) == 'counts as a failure' and automatic(item) == "keeps the reviewers' grade"
    spot = {'queue': 'collection_pilot', 'kind': 'spot_check', 'summary': {'rows': 4}}
    assert outcome(spot, False) == 'removes 4 training rows' and outcome(spot, True) == 'stays in training'
    assert automatic({**spot, 'kind': 'split'}) == 'stays out of training'
    assert sorted(['r10', 'tests', 'r2'], key=order) == ['tests', 'r2', 'r10']


def test_items_appear_while_the_reviewer_is_open(review_campaign):
    queued(review_campaign, 'a')
    model = press(reviewer(review_campaign), 'enter', 'enter')
    queued(review_campaign, 'b')
    assert len(model.items) == 1
    press(model, 'tick')
    assert len(model.items) == 2


class Screen:
    def __init__(self, keys):
        self.keys, self.drawn = list(keys), []

    def getmaxyx(self):
        return 30, 100

    def erase(self):
        self.drawn = []

    def addnstr(self, row, column, text, width, style):
        self.drawn.append(text[:width])

    def refresh(self):
        pass

    def keypad(self, enabled):
        pass

    def timeout(self, milliseconds):
        self.milliseconds = milliseconds

    def get_wch(self):
        key = self.keys.pop(0)
        if key is None:
            raise terminal.curses.error('no input')
        return key


def test_the_terminal_draws_until_the_person_quits(review_campaign, monkeypatch):
    for name in ('curs_set', 'use_default_colors', 'init_pair'):
        monkeypatch.setattr(terminal.curses, name, lambda *args: None)
    monkeypatch.setattr(terminal.curses, 'color_pair', lambda index: index)
    queued(review_campaign, 'a')
    fake = Screen(['\n', None, terminal.curses.KEY_DOWN, 'q'])
    model = reviewer(review_campaign)
    terminal.loop(fake, model)
    assert model.done and fake.milliseconds == terminal.REFRESH_MS and any('Collection (pilot)' in line for line in fake.drawn)
    monkeypatch.setattr(terminal.curses, 'wrapper', lambda function, model: (_ for _ in ()).throw(KeyboardInterrupt))
    terminal.run(review_campaign.review, review_campaign.journal.directory)


@pytest.mark.parametrize('key, name', [(None, 'tick'), ('\n', 'enter'), ('\x1b', 'esc'), ('\t', 'tab'), ('\x7f', 'backspace'),
                                       ('é', 'é'), ('x', 'x'), (terminal.curses.KEY_LEFT, 'left'), (999999, '')])
def test_key_names(key, name):
    assert terminal.key_name(key) == name


def test_copy_uses_a_command_a_clipboard_tool_or_the_terminal(tmp_path, monkeypatch):
    target = tmp_path / 'item.txt'
    monkeypatch.setenv('SHELL', '/bin/sh')
    assert terminal.copy('text', f'cat > {target}') == f'Sent to: cat > {target}' and target.read_text() == 'text'
    assert terminal.copy('text', 'exit 4') == 'The command exited 4.'
    monkeypatch.delenv('SSH_CONNECTION', raising=False)
    monkeypatch.setattr(terminal.shutil, 'which', lambda name: name == 'xclip')
    calls = []
    monkeypatch.setattr(terminal.subprocess, 'run', lambda args, **kwargs: calls.append(args) or type('R', (), {'returncode': 0})())
    assert terminal.copy('text', '') == 'Copied to the clipboard.' and calls == [['xclip', '-selection', 'clipboard']]
    monkeypatch.setattr(terminal.subprocess, 'run', lambda args, **kwargs: type('R', (), {'returncode': 1})())
    assert terminal.copy('text', '') == 'xclip exited 1.'
    monkeypatch.setenv('SSH_CONNECTION', '192.0.2.1 1 192.0.2.2 22')
    written = tmp_path / 'tty'
    monkeypatch.setattr(terminal, 'open', lambda path, mode: written.open(mode), raising=False)
    assert terminal.copy('text', '').startswith('Copied through the terminal.')
    assert written.read_text() == '\033]52;c;dGV4dA==\a'


def test_review_opens_the_reviewer_on_a_terminal(review_campaign, monkeypatch, capsys):
    opened = []
    monkeypatch.setattr('bialy.review.terminal.run', lambda review, campaign: opened.append(campaign))
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    monkeypatch.setattr('sys.stdout.isatty', lambda: True)
    assert cli.main(['review', '--run', str(review_campaign.journal.directory)]) == 0
    assert opened == [review_campaign.journal.directory] and 'No review queues are open yet.' in capsys.readouterr().out


def test_prompts_and_reason_editing_show_their_keys(review_campaign):
    queued(review_campaign, 'a')
    queued(review_campaign, 'b')
    model = press(reviewer(review_campaign), 'enter', 'enter', 'down', 'j', 'up', 'k', '/', 's')
    assert '/s▏   (Enter apply · Esc clear)' in screen(model)
    press(model, 'esc', 'enter', '/')
    assert '(Enter find · Esc cancel)' in screen(model)
    press(model, 'esc', '|')
    assert '(Enter copy · or type a command · Esc cancel)' in screen(model)
    press(model, 'esc', 'v', 'down', 'j', 'up', 'k', 'e')
    assert 'Type a short reason, then press Enter.' in screen(model) and 'Enter done   Backspace delete' in screen(model)
    press(model, 'esc', 'esc', 'esc', 'q')
    assert model.done


def test_a_fully_labelled_pair_reads_labelled(review_campaign):
    review_campaign.task('t1')
    trials = {arm: review_campaign.trial('y-' + arm) for arm in ('base', 'trained')}
    review_campaign.review.open('evaluation', 'evaluation')
    item = review_campaign.review.offer('evaluation', 'pair', [subject('y-' + arm, trials[arm], arm) for arm in trials], [],
                                        {'task_id': 't1', 'family': 'build_test_repair', 'student': 'e4b', 'repetition': 0,
                                         'prompt': 'Add a flag', 'arms': {}}, 0)
    for arm in trials:
        review_campaign.review.label(item, 'episode-y-' + arm, *review_campaign.verdict(), '')
    model = press(reviewer(review_campaign), 'enter', 'enter')
    assert 'labelled' in [line for line in screen(model).splitlines() if 'build_test_repair' in line][0]
