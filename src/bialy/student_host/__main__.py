"""Run pinned student preparation, training, and export operations."""
import argparse
import json
from pathlib import Path

from . import exporting, jobs, materialize, reaper, remote_training, supervision
from .specification import DEFAULT, load, students


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, default=DEFAULT)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('students')
    prepare = commands.add_parser('models')
    prepare.add_argument('--root', type=Path, required=True)
    create = commands.add_parser('create')
    create.add_argument('--student', required=True)
    for name in ('root', 'checkout', 'models', 'splits', 'mirror'):
        create.add_argument('--' + name, type=Path, required=True)
    create.add_argument('--device', default='cuda:0')
    create.add_argument('--pilot', action='store_true')
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
    remote = commands.add_parser('remote')
    remote.add_argument('--configuration', type=Path, required=True)
    remote.add_argument('--resume', type=Path)
    remote.add_argument('--pilot', action='store_true')
    remote.add_argument('--student', required=True)
    for name in ('splits', 'source-key', 'out'):
        remote.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'students':
        print('\n'.join(students(load(args.specification))))
        return
    if getattr(args, 'student', None) is not None and args.student not in students(load(args.specification)):
        parser.error('unknown student %r' % args.student)
    if args.action == 'models':
        result = materialize.models(load(args.specification), args.root)
    elif args.action == 'create':
        result = {'directory': str(jobs.create(args.root, load(args.specification), args.student, args.checkout,
                                               args.models, args.splits, args.mirror, args.device, args.pilot))}
    elif args.action == 'train':
        result = supervision.train(args.directory)
    elif args.action == 'install-reaper':
        result = reaper.install(args.configuration)
    elif args.action == 'reap':
        result = reaper.reap(args.configuration, args.once)
    elif args.action == 'remote':
        result = remote_training.train(args.configuration, args.student, args.splits, args.source_key, args.out, resume=args.resume, pilot=args.pilot)
    else:
        result = exporting.export(args.directory, args.llama, args.revision)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
