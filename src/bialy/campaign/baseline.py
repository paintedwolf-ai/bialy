"""Export the pinned, untrained student checkpoint for paired evaluation.

The base arm differs from the trained one in its weights alone: the host stages the
stock checkpoint with the training tokenizer and chat template, and the same exporter
converts it and serves it against the same rendering check.
"""
import json
import shutil
import subprocess
from pathlib import Path

from ..provenance import file_sha256
from ..student_host import exporting
from ..student_host.jobs import checkpoint, rendering, tree
from .execution import atomic_json
from .journal import CampaignFault, digest


class Baseline:
    def __init__(self, journal, configuration, specification, student, run=subprocess.run):
        self.journal, self.configuration, self.specification, self.student = journal, configuration, specification, student
        self.run = run

    def binding(self):
        cfg = self.configuration
        return {'model_files': checkpoint(self.specification, self.student, cfg['models']),
                'specification': self.specification, 'student': self.student,
                'rendering': rendering(self.specification, cfg['checkout']),
                'host_commit': cfg['host_commit'], 'llama_revision': cfg['llama_revision']}

    def execute(self, inputs, attempt):
        cfg = self.configuration
        binding = self.binding()
        root = self.journal.directory / 'baselines' / self.student / attempt
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'binding.json').exists() and json.loads((root / 'binding.json').read_text()) != binding:
            raise CampaignFault('baseline_inputs_changed')
        atomic_json(root / 'binding.json', binding)
        model = Path(cfg['models']) / self.specification['models'][self.student]['directory']
        pipe = Path(cfg['checkout']) / 'scripts/bialy/custom_model'
        prefix = ['uv', 'run', '--project', str(pipe), '--locked', 'python']
        llama, staged, gguf = Path(cfg['llama']), root / 'checkpoint', root / 'gguf'
        with (root / 'export.log').open('a') as log:
            if not (staged / 'manifest.json').exists():
                # A staging the host did not finish has no manifest and is staged again.
                shutil.rmtree(staged, ignore_errors=True)
                self.run([*prefix, str(pipe / 'base_checkpoint.py'), '--model', str(model), '--out', str(staged)],
                         check=True, stdout=log, stderr=subprocess.STDOUT)
            args = [*prefix, str(pipe / 'gguf_export.py'), '--merge', str(staged), '--llama-cpp', str(llama),
                    '--revision', cfg['llama_revision'], '--quantizer', str(llama / 'build/bin/llama-quantize'),
                    '--server', str(llama / 'build/bin/llama-server'), '--out', str(gguf)]
            if gguf.exists():
                args.append('--resume')
            self.run(args, check=True, stdout=log, stderr=subprocess.STDOUT)
        template = binding['rendering']['template_sha256']
        base, manifest = (json.loads((path / 'manifest.json').read_text()) for path in (staged, gguf))
        if (base.get('source') != binding['model_files'] or base.get('template_sha256') != template
                or base.get('rendering_profile') != binding['rendering']['profile']
                or manifest.get('schema') != exporting.GGUF or manifest.get('llama_cpp_revision') != cfg['llama_revision']
                or manifest.get('merged_manifest_sha256') != file_sha256(staged / 'manifest.json')
                or manifest.get('template_sha256') != template):
            raise CampaignFault('baseline_export_unqualified')
        try:
            exporting.served(manifest, 'quantized', template, None)
        except exporting.JobError as exc:
            raise CampaignFault('baseline_export_unqualified') from exc
        return {'directory': str(gguf), 'files': tree(gguf), 'student': self.student, 'binding_sha256': digest(binding),
                'template_sha256': template}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        binding = self.binding()
        if (output['binding_sha256'] != digest(binding) or tree(output['directory']) != output['files']
                or output['template_sha256'] != binding['rendering']['template_sha256']):
            raise CampaignFault('baseline_artifact_changed')
