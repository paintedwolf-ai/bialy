"""The verdict form: six checks, a receipt citation for each, and an optional reason."""
from ..campaign.grading import DIMENSIONS
from .render import CHECKS


def outcome(item, passed):
    """What saving this verdict does, in plain words."""
    rows = item['summary'].get('rows')
    if item['queue'] == 'evaluation':
        return 'counts as a success' if passed else 'counts as a failure'
    if item['kind'] == 'split':
        return f'admits {rows} training rows' if passed else 'stays out of training'
    return 'stays in training' if passed else f'removes {rows} training rows'


def automatic(item):
    """What a skip leaves in place."""
    if item['queue'] == 'evaluation':
        return "keeps the reviewers' grade"
    return 'stays out of training' if item['kind'] == 'split' else 'stays in training'


class VerdictForm:
    def __init__(self, item, view, decision=None):
        self.item, self.view = item, view
        self.values = dict.fromkeys(DIMENSIONS)
        self.citations = {key: [] for key in DIMENSIONS}
        self.reason = ''
        if decision and decision['decided'] == 'human':
            self.values = dict(decision['dimensions'])
            self.citations = {key: list(value) for key, value in decision['citations'].items()}
            self.reason = decision['reason']
        self.cursor = 0
        self.picker = None
        self.editing = False
        self.message = ''
        self.refs = sorted(view.subject['evidence_ids'], key=lambda identifier: order(view.ref(identifier)))

    @property
    def check(self):
        return DIMENSIONS[self.cursor]

    def result(self):
        if any(value is None for value in self.values.values()):
            return None
        return all(value == 'pass' for value in self.values.values())

    def handle(self, key):
        """Apply one key; returns 'save' or 'cancel' when the form closes."""
        self.message = ''
        if self.editing:
            return self.edit(key)
        if self.picker is not None:
            return self.pick(key)
        if key in DIGITS:
            self.cursor = DIGITS[key]
        elif key in ('up', 'k'):
            self.cursor = max(0, self.cursor - 1)
        elif key in ('down', 'j'):
            self.cursor = min(len(DIMENSIONS) - 1, self.cursor + 1)
        elif key in ('p', 'f'):
            self.values[self.check] = 'pass' if key == 'p' else 'fail'
            self.cursor = min(len(DIMENSIONS) - 1, self.cursor + 1)
        elif key == 'c':
            self.picker = {'cursor': 0, 'selected': set(self.citations[self.check])}
        elif key == 'e':
            self.editing = True
        elif key == 'enter':
            return self.save()
        elif key == 'esc':
            return 'cancel'
        return None

    def save(self):
        for index, key in enumerate(DIMENSIONS):
            if self.values[key] is None:
                self.cursor, self.message = index, 'Set this check to pass or fail.'
                return None
            if not self.citations[key]:
                self.cursor, self.message = index, 'Cite a receipt for this check.'
                return None
        return 'save'

    def pick(self, key):
        picker = self.picker
        if key in ('up', 'k'):
            picker['cursor'] = max(0, picker['cursor'] - 1)
        elif key in ('down', 'j'):
            picker['cursor'] = min(len(self.refs) - 1, picker['cursor'] + 1)
        elif key == ' ':
            picker['selected'] ^= {self.refs[picker['cursor']]}
        elif key == 'enter':
            self.citations[self.check] = sorted(picker['selected'], key=lambda identifier: order(self.view.ref(identifier)))
            self.picker = None
        elif key == 'esc':
            self.picker = None
        return None

    def edit(self, key):
        if key in ('enter', 'esc'):
            self.editing = False
        elif key == 'backspace':
            self.reason = self.reason[:-1]
        elif len(key) == 1 and key.isprintable():
            self.reason += key
        return None

    def instruction(self):
        if self.picker is not None:
            return 'Pick the receipts that show this check. Space toggles.'
        if self.editing:
            return 'Type a short reason, then press Enter.'
        return 'Set each check, cite a receipt, then press Enter.'

    def lines(self):
        out = []
        if self.picker is not None:
            out.append(('  ' + CHECKS[self.check][0], 'bold'))
            for index, identifier in enumerate(self.refs):
                mark = '[x]' if identifier in self.picker['selected'] else '[ ]'
                line = f"{'▸' if index == self.picker['cursor'] else ' '} {mark} {self.view.ref(identifier):<16} {self.label(identifier)}"
                out.append((line, 'cursor' if index == self.picker['cursor'] else 'body'))
            return out
        for index, key in enumerate(DIMENSIONS):
            title, hint = CHECKS[key]
            value = self.values[key]
            shown = {'pass': 'PASS  fail', 'fail': 'pass  FAIL', None: 'pass  fail'}[value]
            cited = ' '.join(self.view.ref(identifier) for identifier in self.citations[key]) or '—'
            line = f"{'▸' if index == self.cursor else ' '} {index + 1} {title:<24} {shown}   cite {cited:<10} {hint}"
            out.append((line, 'cursor' if index == self.cursor else {'pass': 'pass', 'fail': 'fail', None: 'body'}[value]))
        out.append(('  Reason (optional)  ' + self.reason + ('▏' if self.editing else ''), 'body'))
        out.append(('', 'body'))
        result = self.result()
        if result is None:
            out.append(('  Result: set every check to see what saving does', 'hint'))
        else:
            out.append((f"  Result: {'pass' if result else 'fail'} → {outcome(self.item, result)}", 'pass' if result else 'fail'))
        if self.message:
            out.append(('  ' + self.message, 'status'))
        return out

    def label(self, identifier):
        for check in self.view.subject['checks']:
            if check['id'] == identifier:
                return 'check ' + ('passed' if check['passed'] else 'failed')
        invocation = self.view.invocation(identifier)
        return f"{invocation['tool']} {invocation['status']}" if invocation else ''

    def footer(self):
        if self.picker is not None:
            return '↑↓ move   Space toggle   Enter done   Esc cancel'
        if self.editing:
            return 'Enter done   Backspace delete'
        return '1-6 pick   p pass   f fail   c cite   e reason   Enter save   Esc cancel'


DIGITS = {str(index + 1): index for index in range(len(DIMENSIONS))}


def order(ref):
    """Checks first, then invocation references in transcript order."""
    if ref[:1] == 'r' and ref[1:].isdigit():
        return (1, int(ref[1:]), '')
    return (0, 0, ref)
