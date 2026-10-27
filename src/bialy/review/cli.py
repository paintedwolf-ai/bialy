"""bialy review: label a campaign's human review queues.

It reads and writes only files under the campaign directory, so it runs from any shell that
reaches that directory, whether the controller is running, waiting, or stopped. When a queue is
finished, rerun the campaign's resume command to continue.
"""
import json
import sys
import time
from pathlib import Path

from ..campaign.human_review import Review
from ..campaign.journal import CampaignFault
from .render import item_text, queue_title

NOT_WAITING, ERROR, WAITING = 0, 2, 3


def register(sub):
    parser = sub.add_parser('review', help='label a campaign\'s human review queues',
                            description='Label a campaign\'s human review queues. Skipped and unlabelled items '
                                        'keep the automated verdict.')
    parser.add_argument('--run', type=Path, default=Path('.'), help='campaign directory (default: current directory)')
    commands = parser.add_subparsers(dest='review_command')
    status = commands.add_parser('status', help='queues, what is left, and whether the campaign is waiting on you')
    status.add_argument('--json', action='store_true')
    show = commands.add_parser('show', help='print one item as plain text')
    show.add_argument('item')
    show.add_argument('--full', action='store_true', help='print tool output in full')
    skip = commands.add_parser('skip', help='decide an item automatically')
    skip.add_argument('item')
    finish = commands.add_parser('finish', help='close a queue; anything left is decided automatically')
    finish.add_argument('queue')
    finish.add_argument('--yes', action='store_true', help='skip the confirmation')
    wait = commands.add_parser('wait', help='return when every queue the campaign waits on is finished')
    wait.add_argument('--timeout', type=float, help='give up after this many seconds')
    wait.add_argument('--interval', type=float, default=15.0)
    return parser


def run(args):
    if not (args.run / 'campaign.sqlite').is_file():
        print(f'No campaign at {args.run}. Pass --run with a campaign directory.', file=sys.stderr)
        return ERROR
    review = Review(args.run)
    try:
        command = args.review_command or 'status'
        if command == 'status':
            return status(args.run, review, args.json if args.review_command else False)
        if command == 'show':
            print(item_text(review.item(item_id(args.item)), full=args.full))
            return NOT_WAITING
        if command == 'skip':
            review.skip(item_id(args.item))
            print(f'Skipped #{item_id(args.item)}. It is decided automatically.')
            return NOT_WAITING
        if command == 'finish':
            return finish(review, args.queue, args.yes)
        return wait(args.run, review, args.timeout, args.interval)
    except CampaignFault as fault:
        print(message(fault), file=sys.stderr)
        return ERROR


def item_id(text):
    try:
        return int(text.removeprefix('#'))
    except ValueError:
        raise CampaignFault('review_item_missing', details={'item': text}) from None


def message(fault):
    texts = {'review_item_missing': 'No such item: {item}.', 'review_queue_missing': 'No such queue.',
             'review_queue_finished': 'The {queue} queue is finished; its decisions are final.'}
    return texts.get(fault.code, fault.code).format(**fault.details)


def paused(campaign):
    """The controller's pause record, or None while it runs or after it finished."""
    path = campaign / 'waiting.json'
    return json.loads(path.read_text()) if path.exists() else None


def waiting_on_you(campaign, review):
    record = paused(campaign)
    if record is None or record['reason'] != 'human_review' or review.queue(record['queue'])['finished_at']:
        return []
    return [record['queue']]


def summary(campaign, review):
    queues = []
    for queue in review.queues():
        counts = review.counts(queue['name'])
        queues.append({'name': queue['name'], 'title': queue_title(queue['name']),
                       'finished': bool(queue['finished_at']), **counts})
    record = paused(campaign)
    return {'campaign': str(campaign.resolve()), 'queues': queues, 'waiting_on_you': waiting_on_you(campaign, review),
            'resume': record['resume'] if record else None}


def status(campaign, review, as_json):
    state = summary(campaign, review)
    if as_json:
        print(json.dumps(state, indent=2))
        return WAITING if state['waiting_on_you'] else NOT_WAITING
    print('Campaign  ' + state['campaign'])
    if not state['queues']:
        print('No review queues are open yet.')
    for queue in state['queues']:
        progress = f"{queue['open']} left · {queue['labelled']} labelled · {queue['skipped']} skipped"
        print(f"  {queue['title']:<20} {'finished' if queue['finished'] else 'open':<9} {progress}")
    if state['waiting_on_you']:
        names = ', '.join(queue_title(name) for name in state['waiting_on_you'])
        print(f'The campaign is waiting on you: label or skip items, then finish {names}.')
        return WAITING
    if state['resume']:
        print('Reviews are finished. Resume the campaign with:\n  ' + state['resume'])
    else:
        print('The campaign is not waiting on you.')
    return NOT_WAITING


def finish(review, name, confirmed):
    if review.queue(name) is None:
        raise CampaignFault('review_queue_missing')
    left = review.counts(name)['open']
    if not confirmed and sys.stdin.isatty():
        answer = input(f'Finish the {queue_title(name)} queue? {left} items left will be decided automatically. [y/N] ')
        if answer.strip().lower() not in {'y', 'yes'}:
            print('Not finished.')
            return NOT_WAITING
    review.finish(name)
    print(f'Finished {queue_title(name)}. {left} items were decided automatically.')
    return NOT_WAITING


def wait(campaign, review, timeout, interval):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if not waiting_on_you(campaign, review):
            return NOT_WAITING
        if deadline is not None and time.monotonic() >= deadline:
            return WAITING
        time.sleep(interval)
