"""A stand-in backend for the campaign graph: every stage passes unless told otherwise."""
from bialy.campaign import profiles
from bialy.campaign.journal import CampaignFault, StageWaiting


class Backend:
    def __init__(self, failures=(), waiting=(), outputs=None):
        self.failures, self.waiting, self.outputs, self.calls = set(failures), set(waiting), dict(outputs or {}), []

    def execute(self, name, inputs, attempt):
        self.calls.append((name, inputs))
        if name in self.failures:
            raise CampaignFault('fixture_failure')
        if name in self.waiting:
            raise StageWaiting('human_review', {'queue': name, 'left': 1, 'command': 'c'})
        if name in self.outputs:
            return self.outputs[name]
        if name in ('host_qualification', 'student_host_qualification'):
            return {'conservative_reference': {'qualified': True, 'configuration': 'reference'}, 'stage': name}
        if name == 'selector_training':
            return {'policy': {'identity': 'p0'}}
        if name in ('selector_qualification', 'selector_supplied'):
            return {'qualified': True, 'configuration': {'pilot': '/pilot', 'policy': 'p0', 'fallback_rate': 0.1}}
        if name == 'host_adoption':
            return {'host_commit': 'c' * 40, 'host_image': 'image', 'policy': 'p0'}
        if name.startswith('review_'):
            return {'decisions': {}, 'queue': name}
        if name.startswith('compare_'):
            return {'status': 'pass'}
        return {'identity': name}

    def validate(self, name, inputs, output):
        assert output is not None

    def names(self):
        return [name for name, _ in self.calls]

    def inputs(self, name):
        return next(inputs for called, inputs in self.calls if called == name)


def profile(name):
    return profiles.load(name)
