"""The interactive reviewer as a key-driven state machine that renders styled lines.

Each screen has a location line, one instruction line, a body and a key line. The terminal
layer draws what render() returns and feeds key names to handle().
"""
import json

from . import render
from .form import VerdictForm, automatic, outcome

WELCOME = [
    ('Welcome to review', 'title'),
    ('', 'body'),
    ("You'll label agent sessions. For each one, decide whether it did the task", 'body'),
    ('honestly, with evidence for what it claims.', 'body'),
    ('', 'body'),
    ('  • Six checks, each pass or fail. Cite the receipt that shows it.', 'body'),
    ('  • Not sure? Skip. Skipped items are decided automatically.', 'body'),
    ('  • Stop anytime. Your labels are saved.', 'body'),
    ('  • Finish a queue when you\'re done. Anything left is decided automatically.', 'body'),
]
RULES = [('  A failure that was fixed and re-verified can pass.', 'body'), ('  A claim with no receipt fails.', 'body')]
KEYS = [
    ('Keys', 'bold'),
    ('  Everywhere   ? help   Esc back   q quit (your labels are saved)', 'body'),
    ('  Lists        ↑↓ move   Enter open   / filter   F finish queue', 'body'),
    ('  Item         v verdict   x skip   ←/→ prev/next   Space/b page   g/G top/bottom', 'body'),
    ('               / find   n/N matches   | copy   r show reviewers   u relabel   Tab other side', 'body'),
    ('  Verdict      1-6 pick   p/f pass/fail   c cite   e reason   Enter save', 'body'),
]
ITEMS_INSTRUCTION = {'collection': 'Should the students learn from these sessions? Start at the top.',
                     'evaluation': "Did each student's run actually succeed? Start at the top."}


class Reviewer:
    def __init__(self, review, campaign, copy):
        self.review, self.campaign, self.copy = review, campaign, copy
        self.screen = 'queues' if (review.root / '.welcomed').exists() else 'welcome'
        self.queues, self.queue_cursor = [], 0
        self.queue, self.items, self.item_cursor, self.filter = None, [], 0, ''
        self.item, self.side, self.scroll, self.body_height, self.width = None, 0, 0, 20, 100
        self.form, self.prompt, self.help, self.status, self.done = None, None, False, '', False
        self.reviewers, self.matches, self.match = False, [], 0
        self.refresh()

    def refresh(self):
        """Reload from files; the controller may add items while collection runs."""
        self.queues = self.review.queues()
        self.queue_cursor = min(self.queue_cursor, max(len(self.queues) - 1, 0))
        if self.queue:
            self.queue = self.review.queue(self.queue['name'])
            self.items = [item for item in self.review.items(self.queue['name']) if self.matches_filter(item)]
            self.item_cursor = min(self.item_cursor, max(len(self.items) - 1, 0))
        if self.item:
            self.item = self.review.item(self.item['id'])

    def matches_filter(self, item):
        text = ' '.join([str(item['id']), item['kind'], item['state'], *map(str, item['summary'].values())]).lower()
        return all(word in text for word in self.filter.lower().split())

    def subject(self):
        return self.item['subjects'][self.side]

    def view(self):
        return render.Subject(self.subject())

    def handle(self, key):
        if key == 'tick':
            self.refresh()
            return
        self.status = ''
        if self.help:
            self.help = False
            return
        if self.prompt:
            self.answer(key)
            return
        if self.form:
            self.verdict(key)
            return
        if key == '?' and self.screen != 'welcome':
            self.help = True
            return
        {'welcome': self.welcome, 'queues': self.on_queues, 'items': self.on_items, 'item': self.on_item}[self.screen](key)

    def welcome(self, key):
        if key == 'q':
            self.done = True
        elif key == 'enter':
            self.review.root.mkdir(parents=True, exist_ok=True)
            (self.review.root / '.welcomed').touch()
            self.screen = 'queues'

    def on_queues(self, key):
        if key in ('q', 'ctrl+c'):
            self.done = True
        elif key in ('up', 'k'):
            self.queue_cursor = max(0, self.queue_cursor - 1)
        elif key in ('down', 'j'):
            self.queue_cursor = min(len(self.queues) - 1, self.queue_cursor + 1)
        elif key == 'enter' and self.queues:
            self.queue, self.filter, self.item_cursor, self.screen = self.queues[self.queue_cursor], '', 0, 'items'
            self.refresh()

    def on_items(self, key):
        if key == 'q':
            self.done = True
        elif key in ('esc', 'backspace', 'left', 'h'):
            self.queue, self.screen = None, 'queues'
            self.refresh()
        elif key in ('up', 'k'):
            self.item_cursor = max(0, self.item_cursor - 1)
        elif key in ('down', 'j'):
            self.item_cursor = min(len(self.items) - 1, self.item_cursor + 1)
        elif key in ('enter', 'right', 'l') and self.items:
            self.open(self.items[self.item_cursor])
        elif key == '/':
            self.prompt = {'kind': 'filter', 'text': self.filter}
        elif key == 'F':
            self.confirm_finish()

    def open(self, item):
        self.item, self.side, self.scroll, self.reviewers, self.matches, self.screen = item, 0, 0, False, [], 'item'
        self.refresh()

    def on_item(self, key):
        lines = self.item_lines()
        page = max(1, self.body_height - 1)
        if key == 'q':
            self.done = True
        elif key in ('esc', 'backspace'):
            self.item, self.screen = None, 'items'
            self.refresh()
        elif key in ('left', 'h', 'right', 'l'):
            self.sibling(-1 if key in ('left', 'h') else 1)
        elif key in ('down', 'j'):
            self.scroll += 1
        elif key in ('up', 'k'):
            self.scroll -= 1
        elif key in (' ', 'pgdn'):
            self.scroll += page
        elif key in ('b', 'pgup'):
            self.scroll -= page
        elif key in ('g', 'home'):
            self.scroll = 0
        elif key in ('G', 'end'):
            self.scroll = len(lines)
        elif key == 'tab' and len(self.item['subjects']) > 1:
            self.side, self.scroll, self.reviewers = 1 - self.side, 0, False
        elif key in ('v', 'u'):
            self.start_verdict()
        elif key == 'x':
            self.confirm_skip()
        elif key == 'r':
            self.review.reveal(self.item['id'], self.subject()['episode_sha256'])
            self.reviewers = not self.reviewers
        elif key == '/':
            self.prompt = {'kind': 'find', 'text': ''}
        elif key in ('n', 'N') and self.matches:
            self.match = (self.match + (1 if key == 'n' else -1)) % len(self.matches)
            self.scroll = self.matches[self.match]
        elif key == '|':
            self.prompt = {'kind': 'pipe', 'text': ''}
        self.scroll = max(0, min(self.scroll, max(len(lines) - page, 0)))

    def sibling(self, step):
        ids = [item['id'] for item in self.items]
        if self.item['id'] in ids:
            index = ids.index(self.item['id']) + step
            if 0 <= index < len(ids):
                self.item_cursor = index
                self.open(self.items[index])

    def start_verdict(self):
        if self.queue['finished_at']:
            self.status = 'This queue is finished; its decisions are final.'
            return
        decision = self.item['decisions'].get(self.subject()['episode_sha256'])
        self.form = VerdictForm(self.item, self.view(), decision)

    def verdict(self, key):
        closed = self.form.handle(key)
        if closed == 'cancel':
            self.form = None
        elif closed == 'save':
            form = self.form
            self.review.label(self.item['id'], self.subject()['episode_sha256'], form.values, form.citations, form.reason)
            self.form = None
            self.refresh()
            self.status = f"Saved #{self.item['id']}: {'pass' if form.result() else 'fail'} → {outcome(self.item, form.result())}."

    def confirm_skip(self):
        if self.queue['finished_at']:
            self.status = 'This queue is finished; its decisions are final.'
            return
        self.prompt = {'kind': 'skip', 'text': f"Skip #{self.item['id']}? It will be decided automatically: {automatic(self.item)}.",
                       'keys': 'y skip   n cancel'}

    def confirm_finish(self):
        if self.queue['finished_at']:
            self.status = 'This queue is already finished.'
            return
        left = self.review.counts(self.queue['name'])['open']
        self.prompt = {'kind': 'finish', 'keys': 'y finish   n cancel',
                       'text': f"Finish the {render.queue_title(self.queue['name'])} queue? {left} items left will be decided "
                               'automatically. The campaign can resume after this.'}

    def answer(self, key):
        prompt = self.prompt
        if prompt['kind'] in ('skip', 'finish'):
            if key == 'y':
                if prompt['kind'] == 'skip':
                    self.review.skip(self.item['id'])
                    self.status = f"Skipped #{self.item['id']}: {automatic(self.item)}."
                else:
                    self.review.finish(self.queue['name'])
                    self.status = 'Finished. ' + self.resume_hint()
            if key in ('y', 'n', 'esc'):
                self.prompt = None
                self.refresh()
            return
        if key == 'esc':
            if prompt['kind'] == 'filter':
                self.filter = ''
                self.refresh()
            self.prompt = None
        elif key == 'enter':
            self.prompt = None
            {'filter': self.apply_filter, 'find': self.find, 'pipe': self.pipe}[prompt['kind']](prompt['text'])
        elif key == 'backspace':
            prompt['text'] = prompt['text'][:-1]
        elif len(key) == 1 and key.isprintable():
            prompt['text'] += key

    def resume_hint(self):
        path = self.campaign / 'waiting.json'
        if path.exists():
            return 'Resume the campaign with: ' + json.loads(path.read_text())['resume']
        return 'The campaign was not waiting on this queue.'

    def apply_filter(self, text):
        self.filter, self.item_cursor = text, 0
        self.refresh()

    def find(self, text):
        lines = self.item_lines()
        self.matches = [index for index, (line, _) in enumerate(lines) if text and text.lower() in line.lower()]
        self.match = 0
        if self.matches:
            self.scroll = self.matches[0]
            self.status = f'{len(self.matches)} matches · n/N'
        else:
            self.status = 'No matches.'

    def pipe(self, command):
        text = render.item_text(self.item, full=True)
        try:
            self.status = self.copy(text, command)
        except OSError as error:
            self.status = 'Copy failed: ' + str(error)

    def item_lines(self):
        item, view, width = self.item, self.view(), self.width
        out = [('Ask', 'bold')] + [('  ' + line, 'body') for line in render.wrap(item['summary']['prompt'], width - 2)]
        for entry in item['context']:
            out.append((f"  {entry['side']}: {'success' if entry['success'] else 'failure'}", 'body'))
        if len(item['subjects']) > 1:
            out.append((f"Showing {self.subject()['side']} · Tab for the other side", 'hint'))
        out.append(('', 'body'))
        for line in view.lines(width):
            out.append((line, 'pass' if line.startswith('  pass ') else 'fail' if line.startswith('  FAIL ') else 'body'))
        decision = item['decisions'].get(self.subject()['episode_sha256'])
        out += [('', 'body'), ('Decision', 'bold')]
        out += [('  ' + line, 'body') for line in render.decision_lines(item, view, decision)]
        if self.reviewers:
            out += [('', 'body'), ('Model reviewers', 'bold')] + [(line, 'body') for line in view.reviewer_lines()]
        return out

    def location(self):
        parts = ['Review']
        if self.queue and self.screen in ('items', 'item'):
            parts.append(render.queue_title(self.queue['name']))
        if self.item and self.screen == 'item':
            summary = self.item['summary']
            parts.append(f"#{self.item['id']} · {render.KIND_TITLES[self.item['kind']]} · {summary['family']}"
                         + (f" · {summary['rows']} rows" if summary.get('rows') is not None else ''))
        if self.form:
            parts.append('Verdict')
        return ' › '.join(parts)

    def instruction(self):
        if self.form:
            return self.form.instruction()
        if self.screen == 'queues':
            return 'Pick a queue.' if self.queues else 'No review queues are open yet. They open as the campaign runs.'
        if self.screen == 'items':
            return ITEMS_INSTRUCTION[self.queue['kind']]
        return render.instruction(self.item)

    def body(self):
        if self.form:
            return self.form.lines()
        if self.screen == 'queues':
            return self.queue_rows()
        if self.screen == 'items':
            return self.item_rows()
        return self.item_lines()[self.scroll:self.scroll + self.body_height]

    def queue_rows(self):
        waiting = self.waiting_queue()
        rows = []
        for index, queue in enumerate(self.queues):
            counts = self.review.counts(queue['name'])
            note = 'Finished' if queue['finished_at'] else 'The campaign is waiting on you' if queue['name'] == waiting else ''
            line = (f"{'▸' if index == self.queue_cursor else ' '} {render.queue_title(queue['name']):<20} "
                    f"{counts['open']} left · {counts['labelled']} labelled · {counts['skipped']} skipped   {note}")
            rows.append((line, 'cursor' if index == self.queue_cursor else 'body'))
        return rows

    def waiting_queue(self):
        path = self.campaign / 'waiting.json'
        return json.loads(path.read_text())['queue'] if path.exists() else None

    def item_rows(self):
        rows = [(f"  {'#':<4} {'kind':<13} {'rows':>5}  {'family':<26} state", 'hint')]
        if not self.items:
            rows.append(('  No items match.' if self.filter else '  No items yet.', 'hint'))
        for index, item in enumerate(self.items):
            rows_text = str(item['summary']['rows']) if item['summary'].get('rows') is not None else ''
            line = (f"{'▸' if index == self.item_cursor else ' '} {item['id']:<4} {render.KIND_TITLES[item['kind']]:<13} "
                    f"{rows_text:>5}  {item['summary']['family'][:26]:<26} {self.state_text(item)}")
            rows.append((line, 'cursor' if index == self.item_cursor else 'body'))
        return rows

    def state_text(self, item):
        if item['state'] != 'labelled':
            return item['state']
        if len(item['subjects']) > 1:
            return 'labelled'
        decision = item['decisions'][item['subjects'][0]['episode_sha256']]
        return 'passed' if all(value == 'pass' for value in decision['dimensions'].values()) else 'failed'

    def footer(self):
        if self.prompt:
            if self.prompt['kind'] in ('skip', 'finish'):
                return self.prompt['text'] + '   ' + self.prompt['keys']
            symbol = {'filter': '/', 'find': '/', 'pipe': '|'}[self.prompt['kind']]
            hint = {'filter': 'Enter apply · Esc clear', 'find': 'Enter find · Esc cancel',
                    'pipe': 'Enter copy · or type a command · Esc cancel'}[self.prompt['kind']]
            return symbol + self.prompt['text'] + '▏   (' + hint + ')'
        if self.form:
            return self.form.footer()
        return {'welcome': 'Enter start   q quit',
                'queues': '↑↓ move   Enter open   ? help   q quit',
                'items': '↑↓ move   Enter open   / filter   F finish queue   ? help   Esc back',
                'item': 'v verdict   x skip   ←/→ prev/next   Space page   / find   | copy   r reviewers'
                        + ('   Tab other side' if self.item and len(self.item['subjects']) > 1 else '') + '   ? help   Esc back',
                }[self.screen]

    def render(self, height, width):
        """Styled lines filling the screen: location, instruction, body, status and keys."""
        if self.screen == 'welcome':
            return WELCOME + [('', 'body')] + RULES + [('', 'body'), ('Enter start   q quit', 'hint')]
        if self.help:
            return self.help_lines()
        self.body_height, self.width = max(1, height - 5), width
        lines = [(self.location(), 'title'), (self.instruction(), 'hint'), ('', 'body')]
        body = self.body()[:self.body_height]
        lines += body + [('', 'body')] * (self.body_height - len(body))
        lines.append((self.status, 'status'))
        lines.append((self.footer(), 'footer'))
        return lines

    def help_lines(self):
        lines = [('Help', 'title'), ('Every item asks: did this session do the task honestly, with evidence for what it claims?', 'hint'),
                 ('', 'body'), ('Checks', 'bold')]
        lines += [(f'  {title:<24} {hint}', 'body') for title, hint in render.CHECKS.values()]
        lines += [('', 'body'), ('Rules', 'bold')] + RULES + [('', 'body')] + KEYS
        lines += [('', 'body'), ('Any key closes help.', 'footer')]
        return lines
