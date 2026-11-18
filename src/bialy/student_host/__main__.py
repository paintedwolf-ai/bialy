"""Run pinned student preparation, training, and export operations."""
import argparse
import json
import sys
from pathlib import Path

from . import exporting, jobs, machines, materialize, reaper, rehearsal, remote_training, rental_check, status, supervision
from .resources import ResourceError
from .specification import DEFAULT, load, students


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, default=DEFAULT)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('students')
    prepare = commands.add_parser('models')
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--student', action='append', dest='students')
    create = commands.add_parser('create')
    create.add_argument('--student', required=True)
    for name in ('root', 'checkout', 'models', 'splits', 'mirror'):
        create.add_argument('--' + name, type=Path, required=True)
    create.add_argument('--device', default='cuda:0')
    create.add_argument('--schedule', choices=sorted(jobs.SCHEDULES), default='full')
    create.add_argument('--name', help='create the job under this name, or return the one already created under it')
    train = commands.add_parser('train')
    train.add_argument('directory', type=Path)
    export = commands.add_parser('export')
    export.add_argument('directory', type=Path)
    export.add_argument('--llama', type=Path, required=True)
    export.add_argument('--revision', required=True)
    installer = commands.add_parser('install-reaper')
    installer.add_argument('--configuration', type=Path, required=True)
    reap = commands.add_parser('reap')
    reap.add_argument('--configuration', type=Path, required=True)
    reap.add_argument('--once', action='store_true')
    preflight = commands.add_parser('preflight', help='check a machine configuration against a student offline')
    preflight.add_argument('--configuration', type=Path, required=True)
    preflight.add_argument('--student', required=True)
    remote = commands.add_parser('remote', help='train and export on the configured machine; RENTS A MACHINE if the configuration rents')
    remote.add_argument('--configuration', type=Path, required=True)
    remote.add_argument('--resume', type=Path)
    remote.add_argument('--schedule', choices=sorted(jobs.SCHEDULES), default='full')
    remote.add_argument('--student', required=True)
    for name in ('splits', 'source-key', 'out'):
        remote.add_argument('--' + name, type=Path, required=True)
    rehearse = commands.add_parser('rehearse', help='train generated rows, one at the declared length, on the rehearsal schedule and '
                                                    'export; RENTS A MACHINE if the configuration rents; calls no model provider')
    rehearse.add_argument('--configuration', type=Path, required=True)
    rehearse.add_argument('--student', required=True)
    rehearse.add_argument('--out', type=Path, required=True, help='run again with the same directory to take up a stopped rehearsal')
    check = commands.add_parser('rental-check', help='RENTS A MACHINE: prove the rental path of a configuration without training')
    check.add_argument('--configuration', type=Path, required=True)
    check.add_argument('--out', type=Path, required=True)
    check.add_argument('--end', choices=rental_check.MODES, default='release',
                       help='how the rental ends: released by the controller, by the machine after its heartbeats stop, '
                            'or by the reaper after the lease lapses')
    report = commands.add_parser('status', help='where a remote job, rehearsal run or rental check stands')
    report.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    try:
        run(parser, args)
    except (jobs.JobError, ResourceError) as exc:
        # A refusal or a job that did not complete is an outcome with a record, not a crash.
        print(json.dumps({'error': type(exc).__name__, 'diagnostic': str(exc)}), file=sys.stderr)
        sys.exit(1)


def run(parser, args):
    if args.action == 'students':
        print('\n'.join(students(load(args.specification))))
        return
    if getattr(args, 'student', None) is not None and args.student not in students(load(args.specification)):
        parser.error('unknown student %r' % args.student)
    if args.action == 'models':
        result = materialize.models(load(args.specification), args.root, students=args.students)
    elif args.action == 'create':
        result = {'directory': str(jobs.create(args.root, load(args.specification), args.student, args.checkout,
                                               args.models, args.splits, args.mirror, args.device, args.schedule, name=args.name))}
    elif args.action == 'train':
        result = supervision.train(args.directory)
    elif args.action == 'install-reaper':
        result = reaper.install(args.configuration)
    elif args.action == 'reap':
        result = reaper.reap(args.configuration, args.once)
    elif args.action == 'preflight':
        result = machines.preflight(json.loads(args.configuration.read_text()), args.student)
    elif args.action == 'remote':
        result = remote_training.train(args.configuration, args.student, args.splits, args.source_key, args.out,
                                       resume=args.resume, schedule=args.schedule)
    elif args.action == 'rehearse':
        result = rehearsal.rehearse(args.configuration, args.student, args.out)
    elif args.action == 'rental-check':
        result = rental_check.check(args.configuration, args.out, args.end)
    elif args.action == 'status':
        result = status.status(args.directory)
    else:
        result = exporting.export(args.directory, args.llama, args.revision)
    print(json.dumps(result, indent=2))
    if args.action == 'rental-check' and not result['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
