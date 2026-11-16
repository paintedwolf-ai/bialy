"""A selector's trainers, host scripts and engine, faked, over a small shared split."""
import json
from pathlib import Path

from bialy.campaign import ancestry, selector
from bialy.campaign.journal import Journal

DECISIONS = {'version': 3,
             'turn': {'tools': {'load_at': 0.6}, 'skills': {'preload_at': 2.8, 'margin': 0.3}},
             'request': {'load_at': 1.8, 'max_loads': 4, 'nearest_loads': 0},
             'lookup': {'read_at': 2.8, 'margin': 0.3},
             'tool_event': {'read_at': 2.8, 'pointer_at': 2.4, 'margin': 0.3}}
CATALOG = {'state': {'head_tokens': 512}, 'questions': {'tools': {'load_at': 0.6}, 'skills': {'preload_at': 2.8, 'margin': 0.3}},
           'request': {'load_at': 1.8, 'max_loads': 4, 'nearest_loads': 0}, 'lookup': {'read_at': 2.8, 'margin': 0.3}}
CALIBRATED = {'turn': {'tools': {'load_at': 0.55}}, 'request': {'load_at': 2.1, 'max_loads': 3, 'nearest_loads': 1}}
REPLAY = {'overall': {'tools': {'_all': {'recall': 0.9, 'precision': 0.7, 'f1': 0.79, 'macro_recall': 0.5, 'loads_per_turn': 2.1}},
                      'requests': {'wanted': {'top1': 0.6}, 'mrr': 0.6}}}
RERANK = {'pairs': 240, 'lexical': {'mrr': 0.41, 'hit_1': 0.3}, 'blended': {'mrr': 0.52, 'hit_1': 0.4}, 'improved': 60, 'regressed': 12,
          'mrr_gain': 0.11}
PARITY = {'passed': True, 'tolerance': 0.001, 'summary': 'parity: worst gap 0.0002 over 24 turns', 'worst_gap': 0.0002, 'turns': 24}
PRELOAD = {'catalog_revision': 'c0ffee', 'encoding': 'independent', 'option_words': 60, 'max_len': 1024, 'head_tokens': 512,
           'options': {'command': 'Run a command', 'edit': 'Edit a file'}}
REPOSITORIES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa']
RECIPE_HEAD = {'B5': 'turn-load', 'E4': 'unit-rank', 'code-rank': 'code-rank'}


class Run:
    def __init__(self, root):
        self.root = root

    def path(self, *parts):
        return self.root.joinpath(*parts)


class Workbench:
    """The trainers, the host's scripts and its engine, faked: what each was given is kept."""

    def __init__(self, root, replay=REPLAY, rerank=RERANK, calibrated=CALIBRATED, parity=PARITY, preload=PRELOAD, skip=None):
        self.root, self.replay_report, self.rerank_measure, self.calibrated = root, replay, rerank, calibrated
        self.parity_report, self.vocabulary, self.skip = parity, preload, skip
        self.trained, self.replayed, self.reranked, self.probed, self.calibrated_on = [], [], [], [], []
        root.mkdir(parents=True, exist_ok=True)
        (root / 'corpus.json').write_text(json.dumps(CATALOG))
        (root / 'backbone').mkdir(exist_ok=True)
        (root / 'backbone' / 'model.safetensors').write_text('weights')

    def inputs(self, split):
        return {'train': split / 'train.jsonl', 'val': split / 'val.jsonl'}

    def train(self, recipe, inputs, heads_dir, label):
        self.trained.append((recipe, inputs, label))
        if recipe == self.skip:
            return {'skipped': 'no rows'}
        heads_dir.mkdir(parents=True, exist_ok=True)
        head = heads_dir / (RECIPE_HEAD[recipe] + '.safetensors')
        header = json.dumps({'__metadata__': {'label': 'p0-' + recipe, 'backbone': 'enc', 'seed': '11'}}).encode()
        head.write_bytes(len(header).to_bytes(8, 'little') + header)
        return {'head': str(head)}

    def launcher(self, heads, path):
        path.write_text(json.dumps(heads))
        return path

    def calibrate(self, launcher, rows):
        self.calibrated_on.append(rows.name)
        return self.calibrated

    def replay(self, launcher, corpus, rows, out, truth='consensus'):
        self.replayed.append((rows.name, truth))
        self.seen_corpus = json.loads(Path(corpus).read_text())
        Path(out).write_text(json.dumps(self.replay_report))
        return self.replay_report

    def parity(self, launcher, corpus, head, rows, turns, tolerance):
        self.probed.append((Path(head).name, rows.name, turns, tolerance))
        return self.parity_report

    def rerank(self, head, partition, out_dir):
        self.reranked.append((Path(head).name, partition))
        return self.rerank_measure

    def preload(self):
        return self.vocabulary

    def decisions(self, config):
        return DECISIONS

    def catalog(self):
        return self.root / 'corpus.json'

    def backbone(self):
        return self.root / 'backbone'

    def pilot(self, heads, values, preload, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / 'served.json').write_text(json.dumps({'heads': sorted(heads), 'thresholds': values, 'preload': preload}))
        return out


def rows(assignment):
    return {'rows': {'assignment_sha256': assignment.sha256}}


def campaign(tmp_path, **changes):
    graph = ancestry.Graph()
    for repository in REPOSITORIES:
        graph.add('repository:' + repository, 'task', source_class='synthetic')
    assignment = ancestry.allocate(graph, 7, {'train': 0.4, 'validation': 0.3, 'development': 0.3},
                                   {'train': 2, 'validation': 2, 'development': 2})
    run = Run(tmp_path / 'run')
    assignment.save(run.path('ancestry.json'))
    for name in ('train.jsonl', 'val.jsonl', 'development.jsonl'):
        run.path('split').mkdir(parents=True, exist_ok=True)
        run.path('split', name).write_text('{}\n')
    journal = Journal(tmp_path / 'campaign', {})
    workbench = Workbench(tmp_path / 'bench', **changes)
    return journal, assignment, run, workbench


def qualified(tmp_path, **changes):
    journal, assignment, run, workbench = campaign(tmp_path, **changes)
    trained = selector.SelectorTraining(journal, run, workbench).execute(rows(assignment), 'a')
    return journal, run, workbench, trained
