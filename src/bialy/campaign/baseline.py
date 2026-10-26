"""Export the pinned, unchanged student checkpoint for paired evaluation."""
import json
import shutil
import subprocess
from pathlib import Path

from ..provenance import file_sha256
from ..student_host.jobs import checkpoint, tree
from .execution import atomic_json
from .journal import CampaignFault, digest


class Baseline:
    def __init__(self, journal, configuration, specification, student, run=subprocess.run):
        self.journal, self.configuration, self.specification, self.student = journal, configuration, specification, student
        self.run = run

    def execute(self, inputs, attempt):
        cfg = self.configuration
        files = checkpoint(self.specification, self.student, cfg['models'])
        root = self.journal.directory / 'baselines' / self.student / attempt
        staged = root / 'checkpoint'
        model = Path(cfg['models']) / self.specification['models'][self.student]['directory']
        binding = {'model_files': files, 'specification': self.specification, 'student': self.student,
                   'host_commit': cfg['host_commit'], 'llama_revision': cfg['llama_revision']}
        if (root / 'binding.json').exists() and json.loads((root / 'binding.json').read_text()) != binding:
            raise CampaignFault('baseline_inputs_changed')
        atomic_json(root / 'binding.json', binding)
        for name, checksum in files.items():
            target = staged / 'model' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(model / name, target)
            if file_sha256(target) != checksum:
                raise CampaignFault('baseline_checkpoint_changed')
        atomic_json(staged / 'manifest.json', {'schema': 'pw-base-checkpoint/1', 'files': files, 'binding_sha256': digest(binding)})
        pipe = Path(cfg['checkout']) / 'scripts/bialy/custom_model'
        llama = Path(cfg['llama'])
        gguf = root / 'gguf'
        args = ['uv', 'run', '--project', str(pipe), '--locked', 'python', str(pipe / 'gguf_export.py'),
                '--merge', str(staged), '--llama-cpp', str(llama), '--revision', cfg['llama_revision'],
                '--quantizer', str(llama / 'build/bin/llama-quantize'), '--loader', str(llama / 'build/bin/llama-completion'), '--out', str(gguf)]
        if gguf.exists():
            args.append('--resume')
        with (root / 'export.log').open('a') as log:
            self.run(args, check=True, stdout=log, stderr=subprocess.STDOUT)
        manifest = json.loads((gguf / 'manifest.json').read_text())
        if (manifest.get('load_probe') != 'passed' or manifest.get('llama_cpp_revision') != cfg['llama_revision']
                or manifest.get('merged_manifest_sha256') != file_sha256(staged / 'manifest.json')):
            raise CampaignFault('baseline_export_unqualified')
        return {'directory': str(gguf), 'files': tree(gguf), 'student': self.student, 'binding_sha256': digest(binding)}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        binding = {'model_files': checkpoint(self.specification, self.student, self.configuration['models']),
                   'specification': self.specification, 'student': self.student,
                   'host_commit': self.configuration['host_commit'], 'llama_revision': self.configuration['llama_revision']}
        if output['binding_sha256'] != digest(binding) or tree(output['directory']) != output['files']:
            raise CampaignFault('baseline_artifact_changed')
