"""Campaign stages that wrap the host, the head-row pass, the adoption boundary and cleanup."""
import json
import os
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import quote

from .. import faults, preflight
from .. import run as runmod
from ..provenance import file_sha256
from . import adoption, reviewer_controls
from .evaluation import Evaluation
from .journal import CampaignFault, StageWaiting, digest
from .orchestration import BOUNDARY
from .preflight import inspect
from .serving import Serving


def prepare_fleet(root):
    if os.geteuid() != 0:
        raise CampaignFault('campaign_requires_fleet_mount_privileges')
    for name in ('cache', 'repos'):
        (Path(root) / name).mkdir(parents=True, exist_ok=True)


def read_journal(directory):
    """Another campaign's journal, read-only."""
    path = Path(directory) / 'campaign.sqlite'
    connection = sqlite3.connect('file:' + quote(str(path.resolve()), safe='/') + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def passed_output(source, name):
    row = source.execute('SELECT status,output FROM stages WHERE name=?', (name,)).fetchone()
    if row is None or row['status'] != 'passed':
        raise CampaignFault('source_stage_unqualified', details={'stage': name})
    return json.loads(row['output'])


class Host:
    """The pinned host image and its preflight on one commit."""

    def __init__(self, run, configuration, teachers):
        self.run, self.configuration, self.teachers = run, configuration, teachers

    def execute(self, inputs, attempt):
        cfg = self.configuration
        source = inspect(self.run.checkout, cfg['host_commit'], self.run.image('off'), self.run.path('host-contract'))
        prepare_fleet(cfg['factory_root'])
        image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', self.run.image('off')], text=True))[0]
        if image['Config'].get('Labels', {}).get('pw.host_commit') != cfg['host_commit']:
            raise CampaignFault('campaign_runner_source_mismatch')
        path = self.run.path('preflight-off', 'report.json')
        try:
            report = json.loads(path.read_text()) if path.exists() else preflight.qualify(self.run, 'off')
        except faults.RunFault as exc:
            raise CampaignFault(exc.code, details={'shard': exc.shard}) from exc
        if report['image_id'] != image['Id'] or report['settings_sha256'] != digest(self.run.settings):
            raise CampaignFault('campaign_host_preflight_changed')
        return {'image': image['Id'], 'host_commit': cfg['host_commit'], 'source': source, 'preflight_sha256': file_sha256(path),
                'conservative_reference': {'qualified': True, 'mode': 'explicit_tool_loading'},
                'teacher_capture_qualified': {teacher['id']: teacher['id'] == self.run.factory.generators()[0].id
                                              for teacher in self.teachers}}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if self.execute(inputs, None) != output:
            raise CampaignFault('campaign_host_evidence_changed')


class ReviewerQualification:
    """Qualify every reviewer slot on the versioned controls before any episode is reviewed."""

    def __init__(self, reviewers):
        self.reviewers = reviewers

    def execute(self, inputs, attempt):
        return {'slots': reviewer_controls.qualify(self.reviewers)}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output != self.execute(inputs, None):
            raise CampaignFault('reviewer_qualification_changed')


class ServedEvaluation:
    """Application sessions against one exported artifact served for the student."""

    def __init__(self, journal, lineage, driver, specification, configuration, student, arm):
        self.journal, self.driver, self.specification, self.configuration = journal, driver, specification, configuration
        self.student, self.arm = student, arm
        self.actor = {'kind': 'student', 'id': student, 'family': 'gemma',
                      'provider': 'student-' + student, 'model': specification['models'][student]['directory']}
        self.evaluation = Evaluation(journal, lineage, driver, student, arm, self.actor)

    def execute(self, inputs, attempt):
        artifact = inputs['artifact']
        model = Path(artifact['directory']) / 'model-Q4_K_M.gguf'
        server = Serving(self.journal.directory / 'serving' / self.student / self.arm / attempt, self.configuration)
        with server.session(model, artifact['files']['model-Q4_K_M.gguf'], self.specification['context'], self.actor['model']) as url:
            surface = self.driver.factory_run.factory.run['custom_model']['students'][self.student]
            previous = surface['base_url']
            try:
                surface['base_url'] = url
                return self.evaluation.execute(inputs, attempt)
            finally:
                surface['base_url'] = previous

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        self.evaluation.validate(inputs, output)


class HeadRows:
    """The head-row pass on C0: the existing run pipeline through its splits and site dumps."""

    def __init__(self, run, execute=runmod.execute):
        self.run, self.run_stages = run, execute

    def execute(self, inputs, attempt):
        try:
            self.run_stages(self.run, until='coderank')
        except runmod.RunError as exc:
            raise CampaignFault('head_rows_failed', details={'error': str(exc)[-2000:]}) from exc
        return self.facts()

    def facts(self):
        ancestry = self.run.path('ancestry.json')
        if not ancestry.exists():
            raise CampaignFault('head_rows_incomplete')
        return {'run': self.run.name, 'assignment_sha256': json.loads(ancestry.read_text())['assignment_sha256']}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if self.facts() != output:
            raise CampaignFault('head_rows_changed')


class SuppliedSelector:
    """A P0 another campaign qualified, read from that campaign's journal."""

    def __init__(self, directory):
        self.directory = Path(directory) if directory else None

    def execute(self, inputs, attempt):
        if self.directory is None:
            return {'qualified': False}
        with read_journal(self.directory) as source:
            qualification = passed_output(source, 'selector_qualification')
        pilot = Path(qualification['configuration']['pilot'])
        if not (pilot / 'heads').is_dir() or not (pilot / 'overlay' / 'config').is_dir():
            raise CampaignFault('supplied_selector_incomplete')
        return {'qualified': True, 'source': str(self.directory), 'configuration': qualification['configuration']}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output != self.execute(inputs, None):
            raise CampaignFault('supplied_selector_changed')


class Adoption:
    """The boundary between segments: waits for an adopted host commit whose decision
    configuration equals P0, and refuses any other."""

    def __init__(self, checkout, request, command, process=subprocess.run):
        self.checkout, self.request, self.command, self.process = checkout, request, command, process

    def execute(self, inputs, attempt):
        policy = inputs['policy']
        if self.request is None:
            raise StageWaiting(BOUNDARY, {'policy': policy['identity'], 'command': self.command})
        try:
            facts = adoption.check(self.checkout, self.request['host_commit'], policy, self.process)
        except CampaignFault as exc:
            raise StageWaiting(BOUNDARY, {'policy': policy['identity'], 'command': self.command, 'refused': exc.record()}) from exc
        return {**facts, 'host_image': self.request['host_image']}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if adoption.check(self.checkout, output['host_commit'], inputs['policy'], self.process)['policy'] != output['policy']:
            raise CampaignFault('host_adoption_changed')


class Cleanup:
    """Reap every training resource lease the campaign's remote jobs registered."""

    def __init__(self, configuration, reap=None):
        self.configuration, self.reap = configuration, reap

    def execute(self, inputs, attempt):
        remote = self.configuration.get('remote')
        if self.configuration['full_training'] != 'remote' or not remote:
            return {'reaped': False, 'reason': 'local_training'}
        from ..student_host import reaper
        (self.reap or reaper.reap)(remote, once=True)
        return {'reaped': True, 'configuration_sha256': file_sha256(remote)}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output.get('reaped') and output['configuration_sha256'] != file_sha256(self.configuration['remote']):
            raise CampaignFault('cleanup_configuration_changed')
