"""Small synthetic tasks for exercising the factory's complete data path."""
import json
from pathlib import Path

from ..provenance import file_sha256
from .execution import atomic_json
from .journal import CampaignFault
from .lineage import FAMILIES


def examples(number):
    token = 'batch-' + str(number)
    return {
        'implementation_investigation': (
            'Inspect pricing.py and its callers. Fix the charge so orders at or above the documented threshold receive the discount. '
            'Add boundary tests, run them, and explain the cause with evidence.',
            {'pricing.py': f'THRESHOLD = {number}\ndef charge(n):\n    return n * (2 if n > THRESHOLD else 3)\n',
             'app.py': 'from pricing import charge\nprint(charge(4))\n'},
            f'from pricing import charge\nassert charge({number - 1}) == {(number - 1) * 3}\nassert charge({number}) == {number * 2}\n'),
        'missing_path_or_executable': (
            'The documented verification command is python3 verify.py. Try it, investigate any failure, locate the actual checks, '
            f'and fix compute(n) to return n + {number - 3}. Report the verification command that worked.',
            {'value.py': 'def compute(n):\n    return n\n',
             'checks/check.py': f'import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]))\n'
                                f'from value import compute\nassert compute(3) == {number}\n'},
            f'from value import compute\nassert compute(3) == {number}\nassert compute(0) == {number - 3}\nassert compute(-2) == {number - 5}\n'),
        'interactive_capture': (
            f'Run dialog.py in an interactive terminal and answer its prompt with {token}. '
            'Wait for it to finish and verify confirmation.txt. Do not change the program or write the confirmation file yourself.',
            {'dialog.py': "from pathlib import Path\nimport sys\nassert sys.stdin.isatty()\n"
                          "answer = input('Confirmation code: ')\nPath('confirmation.txt').write_text(answer+'\\n')\n"},
            f"assert Path('confirmation.txt').read_text() == {token + chr(10)!r}\n"
            "assert any(i['tool'] == 'terminal_send' and i['invoked'] for s in facts['sessions'] for i in s['invocations'])\n"),
        'build_test_repair': (
            'Run the unit tests in test_math.py, fix the failing calculation without changing the tests, and verify the repair.',
            {'maths.py': 'def total(items):\n    return len(items)\n',
             'test_math.py': f'import unittest\nfrom maths import total\nclass Checks(unittest.TestCase):\n'
                             f'    def test_total(self):\n        self.assertEqual(total([2,{number}]),{number + 2})\n'
                             "if __name__ == '__main__': unittest.main()\n"},
            f'from maths import total\nassert total([2,{number}]) == {number + 2}\nassert total([]) == 0\n'),
        'partial_effect_recovery': (
            'Run python3 stage.py once to prepare the output. If it fails, inspect what it already changed before recovering. '
            'Finish the result without repeating a completed append. Verify that the journal contains exactly one entry.',
            {'stage.py': f"from pathlib import Path\nwith Path('journal.txt').open('a') as f: f.write({token + chr(10)!r})\n"
                         "raise RuntimeError('publication failed after journal append')\n"},
            f"assert Path('journal.txt').read_text() == {token + chr(10)!r}\nassert Path('result.txt').read_text().strip() == {token!r}\n"),
        'selection_miss': (
            'Inspect payload.json and create summary.json containing its item count and sum. Find the capabilities you need, '
            'then verify the generated JSON against the input.',
            {'payload.json': json.dumps({'items': [number, 2, 3]}) + '\n'},
            f"assert json.loads(Path('summary.json').read_text()) == {{'count':3,'sum':{number + 5}}}\n"),
        'long_history': (
            'Inspect requirements.txt. Implement settings.json with those requirements and verify it; subsequent requests will extend it.',
            {'requirements.txt': f'Keep code {token!r} and enabled=true in settings.json.\n'},
            f"assert json.loads(Path('settings.json').read_text()) == {{'code':{token!r},'enabled':True,'retries':3,'mode':'strict'}}\n"),
        'structured_closeout': (
            f'Create manifest.json with name {token!r}, version 1, and an empty files list. Verify its exact contents. '
            'Finish with a concise account of what changed and the verification you actually ran.',
            {}, f"assert json.loads(Path('manifest.json').read_text()) == {{'name':{token!r},'version':1,'files':[]}}\n"),
    }


def write_exact(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != body:
        raise CampaignFault('pilot_bank_changed')
    path.write_text(body)


def build(directory):
    directory = Path(directory)
    tasks = []
    for index, partition in enumerate(('train', 'validation', 'development', 'system_test')):
        for family_index, family in enumerate(FAMILIES):
            identifier = partition + '-' + family
            prompt, fixtures, assertions = examples(11 + index * 17 + family_index)[family]
            if family == 'partial_effect_recovery':
                prompt += ' Write the journal entry without its newline to result.txt as the final result.'
            root = directory / identifier
            for name, body in fixtures.items():
                write_exact(root / 'fixtures' / name, body)
            script = "import json,sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path.cwd()))\nfacts=json.loads(Path('/evidence/host-evidence.json').read_text())\n" + assertions
            write_exact(root / 'check.py', script)
            files = {str(path.relative_to(root)): file_sha256(path) for path in root.rglob('*') if path.is_file()}
            task = {'id': identifier, 'partition': partition, 'family': family, 'source_class': 'synthetic',
                    'workspace': 'python-cli', 'project_path': '/work/python-cli', 'prompt': prompt,
                    'oracle_directory': identifier, 'oracle_files': files,
                    'fixture_files': {name: 'fixtures/' + name for name in fixtures},
                    'allowed_write_roots': ['.'], 'artifact_paths': [],
                    'checks': [{'id': 'artifact', 'argv': ['python3', '/oracle/check.py']}],
                    'required_checks': ['artifact', 'host-closeout'], 'prompt_timeout': '10m'}
            if family == 'long_history':
                task['follow_ups'] = [{'prompt': 'Add retries=3 while retaining all existing requirements and verify the file.'},
                                      {'prompt': 'Add mode="strict", retain the original code and every earlier requirement, and verify the result.'}]
            tasks.append(task)
    atomic_json(directory / 'tasks.json', tasks)
    return tasks
