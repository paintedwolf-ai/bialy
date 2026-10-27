"""Human review queues: items a person labels, skips, or leaves to the automated verdict.

Review state is plain files under <campaign>/review, so a person can review from any shell
that reaches the campaign directory while the controller runs, waits, or is stopped:

    <queue>/queue.json                 opened by the controller
    <queue>/items/<id>.json            written once by the controller
    <queue>/decisions/<id>/<n>.json    decision receipts, appended by the reviewer
    <queue>/revealed/<id>/<episode>    written when the reviewer shows the model reviewers' verdicts
    <queue>/finished.json              written once by the reviewer; lists the final receipts

Each episode belongs to at most one item. A skip, and every item without a receipt when its
queue finishes, keeps the automated verdict. Gates freeze a finished queue's decisions into
the inputs of the stages that use them.
"""
import datetime
import json
import os
import uuid
from pathlib import Path

from .grading import DIMENSIONS, reviewed
from .journal import CampaignFault, StageWaiting, canonical, digest

QUEUE_KINDS = ('collection', 'evaluation')
ITEM_ORDER = {'split': 0, 'pair': 1, 'spot_check': 1}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def write(path, value, exclusive=False):
    """Publish a JSON file atomically; an exclusive write fails if the file already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path):
    return json.loads(path.read_text())


class Review:
    def __init__(self, campaign):
        self.root = Path(campaign) / 'review'

    def path(self, name):
        return self.root / name

    def open(self, name, kind):
        if kind not in QUEUE_KINDS:
            raise CampaignFault('review_queue_kind_invalid')
        path = self.path(name) / 'queue.json'
        if not path.exists():
            write(path, {'schema': 'pw-review-queue/1', 'name': name, 'kind': kind, 'opened_at': now()})
        queue = self.queue(name)
        if queue['kind'] != kind:
            raise CampaignFault('review_queue_kind_changed')
        return queue

    def queue(self, name):
        path = self.path(name) / 'queue.json'
        if not path.exists():
            return None
        finished = self.path(name) / 'finished.json'
        return {**read(path), 'finished_at': read(finished)['finished_at'] if finished.exists() else None}

    def queues(self):
        names = sorted(path.parent.name for path in self.root.glob('*/queue.json'))
        return sorted((self.queue(name) for name in names), key=lambda queue: (queue['opened_at'], queue['name']))

    def records(self):
        return [read(path) for path in self.root.glob('*/items/*.json')]

    def offer(self, queue, kind, subjects, context, summary, priority):
        """Add an item for subjects no other item covers; returns its id, or None when nothing is new."""
        if kind not in ITEM_ORDER:
            raise CampaignFault('review_item_kind_invalid')
        if self.queue(queue) is None:
            raise CampaignFault('review_queue_missing')
        records = self.records()
        covered = {subject['episode_sha256'] for record in records for subject in record['subjects']}
        fresh = [subject for subject in subjects if subject['episode_sha256'] not in covered]
        if not fresh:
            return None
        identifier = max((record['id'] for record in records), default=0) + 1
        write(self.path(queue) / 'items' / f'{identifier}.json',
              {'schema': 'pw-review-item/1', 'id': identifier, 'queue': queue, 'kind': kind, 'priority': priority,
               'subjects': fresh, 'context': context, 'summary': summary}, exclusive=True)
        return identifier

    def item(self, identifier):
        paths = list(self.root.glob(f'*/items/{identifier}.json'))
        if len(paths) != 1:
            raise CampaignFault('review_item_missing', details={'item': identifier})
        item = read(paths[0])
        item['decisions'] = self.latest(item)
        item['state'] = self.state(item)
        return item

    def items(self, queue):
        items = [self.item(read(path)['id']) for path in self.path(queue).glob('items/*.json')]
        return sorted(items, key=lambda item: (ITEM_ORDER[item['kind']], -item['priority'], item['id']))

    def receipts(self, item):
        directory = self.path(item['queue']) / 'decisions' / str(item['id'])
        return sorted(directory.glob('*.json'), key=lambda path: int(path.stem))

    def latest(self, item, receipts=None):
        decisions = {}
        for path in self.receipts(item) if receipts is None else receipts:
            decision = read(path)
            decisions[decision['episode_sha256']] = decision
        return decisions

    def state(self, item):
        decisions = [item['decisions'].get(subject['episode_sha256']) for subject in item['subjects']]
        if all(decision and decision['decided'] == 'human' for decision in decisions):
            return 'labelled'
        if any(decision and decision['decided'] == 'automatic' for decision in decisions):
            return 'skipped'
        return 'automatic' if self.queue(item['queue'])['finished_at'] else 'open'

    def subject(self, item, episode_sha256):
        subject = next((subject for subject in item['subjects'] if subject['episode_sha256'] == episode_sha256), None)
        if subject is None:
            raise CampaignFault('review_subject_missing')
        return subject

    def reveal(self, identifier, episode_sha256):
        """Record that the model reviewers' verdicts were shown before this subject was labelled."""
        item = self.item(identifier)
        self.subject(item, episode_sha256)
        path = self.path(item['queue']) / 'revealed' / str(identifier) / episode_sha256
        if not path.exists():
            write(path, {'at': now()})

    def revealed(self, item, episode_sha256):
        return (self.path(item['queue']) / 'revealed' / str(item['id']) / episode_sha256).exists()

    def label(self, identifier, episode_sha256, dimensions, citations, reason):
        item = self.item(identifier)
        subject = self.subject(item, episode_sha256)
        if set(dimensions) != set(DIMENSIONS) or any(value not in {'pass', 'fail'} for value in dimensions.values()):
            raise CampaignFault('review_dimensions_invalid')
        evidence = set(subject['evidence_ids'])
        if set(citations) != set(DIMENSIONS) or any(not cited or not set(cited) <= evidence for cited in citations.values()):
            raise CampaignFault('review_citations_invalid')
        if not isinstance(reason, str):
            raise CampaignFault('review_decision_invalid')
        self.record(item, [episode_sha256], {'decided': 'human', 'dimensions': dimensions,
                                             'citations': {key: sorted(set(value)) for key, value in citations.items()},
                                             'reason': reason.strip(), 'revealed_reviewers': self.revealed(item, episode_sha256)})

    def skip(self, identifier):
        item = self.item(identifier)
        self.record(item, [subject['episode_sha256'] for subject in item['subjects']], {'decided': 'automatic'})

    def record(self, item, episodes, decision):
        if self.queue(item['queue'])['finished_at']:
            raise CampaignFault('review_queue_finished', details={'queue': item['queue']})
        directory = self.path(item['queue']) / 'decisions' / str(item['id'])
        for episode in episodes:
            body = {'schema': 'pw-human-review-decision/1', 'queue': item['queue'], 'item': item['id'],
                    'episode_sha256': episode, 'at': now(), **decision}
            while True:
                sequence = max((int(path.stem) for path in directory.glob('*.json')), default=0) + 1
                try:
                    write(directory / f'{sequence}.json', body, exclusive=True)
                    break
                except FileExistsError:
                    continue

    def finish(self, name):
        if self.queue(name) is None:
            raise CampaignFault('review_queue_missing')
        receipts = {str(read(path)['id']): [receipt.name for receipt in self.receipts(read(path))]
                    for path in self.path(name).glob('items/*.json')}
        try:
            write(self.path(name) / 'finished.json', {'schema': 'pw-review-finished/1', 'finished_at': now(),
                                                       'receipts': receipts}, exclusive=True)
        except FileExistsError:
            pass

    def counts(self, name):
        states = [item['state'] for item in self.items(name)]
        return {state: states.count(state) for state in ('open', 'labelled', 'skipped', 'automatic')}

    def frozen(self, name, episodes):
        """Human decisions for these episodes from finished queues, keyed by episode."""
        queue = self.queue(name)
        if queue is None or not queue['finished_at']:
            raise CampaignFault('review_queue_unfinished', details={'queue': name})
        owners = {subject['episode_sha256']: record for record in self.records() for subject in record['subjects']}
        decisions = {}
        for episode in episodes:
            record = owners.get(episode)
            if record is None:
                continue
            finished = self.path(record['queue']) / 'finished.json'
            if not finished.exists():
                raise CampaignFault('review_queue_unfinished', details={'queue': record['queue']})
            names = read(finished)['receipts'].get(str(record['id']), [])
            directory = self.path(record['queue']) / 'decisions' / str(record['id'])
            decision = self.latest(record, [directory / name for name in names]).get(episode)
            if decision and decision['decided'] == 'human':
                decisions[episode] = decision
        return {'queue': name, 'finished_at': queue['finished_at'], 'decisions': decisions, 'decisions_sha256': digest(decisions)}


def subject(name, trial, side):
    """What a reviewer needs from a driven episode, so review never reads the campaign journal."""
    return {'episode_sha256': trial['verdict']['episode_sha256'], 'trial': name, 'side': side,
            'directory': trial['directory'], 'evidence_ids': trial['receipt']['evidence_ids'],
            'checks': trial['receipt']['checks'], 'reviews': trial['reviews']}


class Gate:
    """Pause the campaign until a queue is finished, then pass its frozen decisions on."""
    def __init__(self, review, queue, kind, command):
        self.review, self.name, self.kind, self.command = review, queue, kind, command

    def offer(self, inputs):
        """Add the queue's items before checking it; collection offers its items while it runs."""

    def execute(self, inputs, attempt):
        self.review.open(self.name, self.kind)
        self.offer(inputs)
        counts = self.review.counts(self.name)
        if not self.review.queue(self.name)['finished_at']:
            if not counts['open'] and not counts['labelled'] and not counts['skipped']:
                self.review.finish(self.name)
            else:
                raise StageWaiting('human_review', {'queue': self.name, 'left': counts['open'], 'command': self.command})
        return self.review.frozen(self.name, self.episodes(inputs))

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if self.review.frozen(self.name, self.episodes(inputs)) != output:
            raise CampaignFault('review_decisions_changed', details={'queue': self.name})


class CollectionGate(Gate):
    def __init__(self, review, queue, command):
        super().__init__(review, queue, 'collection', command)

    def episodes(self, inputs):
        return [episode['episode_sha256'] for episode in inputs['collection']['episodes']]

    def offer_episode(self, episode, trial, task):
        """Queue an exported collection episode whose label could change the dataset."""
        self.review.open(self.name, self.kind)
        kind = {'pass': 'spot_check', 'split': 'split'}.get(reviewed(trial))
        if kind is None or episode['exports'] is None:
            return None
        rows = min(exported['rows'] for exported in episode['exports'].values())
        summary = {'task_id': task['id'], 'family': task['family'], 'partition': task['partition'],
                   'repetition': episode['repetition'], 'rows': rows, 'prompt': task['prompt']}
        return self.review.offer(self.name, kind, [subject(episode['stage'], trial, None)], [], summary, rows)


class EvaluationGate(Gate):
    def __init__(self, review, queue, journal, command):
        super().__init__(review, queue, 'evaluation', command)
        self.journal = journal

    def attempts(self, inputs):
        return [attempt for student in sorted(inputs['evaluations']) for arm in ('base', 'trained')
                for attempt in inputs['evaluations'][student][arm]['attempts']]

    def episodes(self, inputs):
        return [attempt['episode_sha256'] for attempt in self.attempts(inputs) if attempt['episode_sha256']]

    def subject(self, attempt):
        return subject(attempt['trial'], self.journal.stage(attempt['trial'])['output'], attempt['arm'])

    def summary(self, attempt):
        task = json.loads(self.journal.db.execute('SELECT spec FROM tasks WHERE id=?', (attempt['task_id'],)).fetchone()[0])
        return {'task_id': task['id'], 'family': task['family'], 'student': attempt['student'],
                'repetition': attempt['attempt'], 'prompt': task['prompt']}

    def offer(self, inputs):
        paired = set()
        for student in sorted(inputs['evaluations']):
            arms = {arm: {(row['task_id'], row['attempt']): row for row in inputs['evaluations'][student][arm]['attempts']}
                    for arm in ('base', 'trained')}
            for key in sorted(arms['base']):
                sides = [arms['base'][key], arms['trained'][key]]
                if sides[0]['success'] == sides[1]['success']:
                    continue
                # The successful side passed every check, so at least one side is changeable.
                changeable = [side for side in sides if side['review'] != 'fail']
                context = [{'side': side['arm'], 'trial': side['trial'], 'success': side['success']} for side in sides]
                self.review.offer(self.name, 'pair', [self.subject(side) for side in changeable], context,
                                  {**self.summary(sides[0]), 'arms': {side['arm']: side['success'] for side in sides}}, 0)
                paired.update(side['episode_sha256'] for side in changeable)
        for attempt in self.attempts(inputs):
            if attempt['review'] == 'split' and attempt['episode_sha256'] not in paired:
                self.review.offer(self.name, 'split', [self.subject(attempt)], [],
                                  {**self.summary(attempt), 'arm': attempt['arm']}, 0)
