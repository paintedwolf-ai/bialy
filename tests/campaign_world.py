"""A fake world for driving the campaign runtime offline: providers, rented machines, the
adopted host's files, and handlers that use them the way the real stages do."""
import json

from bialy.campaign.journal import CampaignFault

POLICY = {'identity': 'p0', 'decision_contract': 3, 'heads': {'turn-load': {'sha256': 'a' * 64}},
          'thresholds': {'turn.tools.load_at': 0.55}}


class World:
    def __init__(self):
        self.provider_requests, self.rentals, self.crash, self.fail = [], [], set(), set()

    def request(self, stage, attempt):
        self.provider_requests.append((stage, attempt))

    def rent(self, stage, attempt):
        self.rentals.append((stage, attempt))

    def owned(self, stage, records):
        return [item for item in records if item[0] == stage]


class Disconnect(BaseException):
    """The controller dies mid-stage: nothing in the campaign catches it."""


class Stage:
    """A stage that pays for its work once: collection makes provider requests, training rents
    a machine. After a disconnect, reconcile adopts what the world already holds."""

    def __init__(self, world, name, output):
        self.world, self.name, self.output = world, name, output

    def execute(self, inputs, attempt):
        if self.name.startswith('collection'):
            self.world.request(self.name, attempt)
        if self.name.startswith('train_'):
            self.world.rent(self.name, attempt)
        if self.name in self.world.crash:
            self.world.crash.discard(self.name)
            raise Disconnect(self.name)
        if self.name in self.world.fail:
            raise CampaignFault('fixture_failure')
        return self.output

    def reconcile(self, inputs, attempt):
        held = self.world.owned(self.name, self.world.provider_requests + self.world.rentals)
        if (self.name.startswith(('collection', 'train_'))) and not held:
            raise CampaignFault('execution_outcome_unknown')
        return self.output

    def validate(self, inputs, output):
        assert output == self.output


def outputs(students, selector_qualified=True):
    value = {'host_qualification': {'conservative_reference': {'qualified': True, 'mode': 'explicit_tool_loading'}, 'host': 'c0'},
             'student_host_qualification': {'conservative_reference': {'qualified': True, 'mode': 'explicit_tool_loading'}, 'host': 'c1'},
             'reviewer_qualification': {'slots': 2}, 'teacher_qualification': {'teachers': [], 'focuses': {}},
             'head_rows': {'run': 'heads', 'assignment_sha256': 'e' * 64},
             'selector_training': {'policy': POLICY, 'heads': {}, 'launcher': '/launcher', 'corpus': '/corpus', 'pilot': '/pilot'},
             'selector_qualification': {'qualified': selector_qualified,
                                        'configuration': {'pilot': '/pilot', 'policy': 'p0', 'fallback_rate': 0.1}},
             'selector_holdout': {'schema': 'pw-selector-holdout/1', 'policy': 'p0'},
             'selector_supplied': {'qualified': False},
             'collection': {'episodes': ['shared']}, 'dataset': {'data_status': 'data_ready'}, 'evaluation_plan': {'tasks': []},
             'cleanup': {'reaped': True}}
    for student in students:
        value.update({'baseline_' + student: {'artifact': 'base-' + student}, 'evaluate_base_' + student: {'attempts': []},
                      'train_' + student: {'job': student}, 'export_' + student: {'gguf': student, 'mlx': student, 'lora_adapter': student},
                      'evaluate_trained_' + student: {'attempts': []}, 'compare_' + student: {'status': 'pass'}})
    return value


def git(files):
    """`git show COMMIT:PATH` over a dictionary of commit -> path -> text."""
    class Result:
        def __init__(self, code, stdout):
            self.returncode, self.stdout, self.stderr = code, stdout, ''

    def process(command, **kwargs):
        commit, path = command[-1].split(':', 1)
        body = files.get(commit, {}).get(path)
        return Result(0 if body is not None else 128, body or '')
    return process


def adopted_files(selector_config_path, release_path, load_at):
    return {release_path: json.dumps({'release': 'r1', 'heads': {'turn-load': {'sha256': 'a' * 64}}}),
            selector_config_path: 'version: 3\nturn:\n  tools:\n    load_at: %s\n' % load_at}
