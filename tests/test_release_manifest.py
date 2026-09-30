import importlib.util
import json
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('release_manifest', Path(__file__).parents[1] / 'train-host/release_manifest.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def heads(tmp_path, encoding='joint'):
    for name in ('turn-load', 'unit-rank', 'code-rank'):
        metadata = {'model': 'test-model', 'label': name, 'corpus': 'training-corpus',
                    'max_len': '1024', 'head_max_len': '512', 'tool_encoding': encoding}
        body = json.dumps({'__metadata__': metadata}).encode()
        (tmp_path / (name + '.safetensors')).write_bytes(struct.pack('<Q', len(body)) + body)
    return tmp_path


def corpus():
    return {'catalog_revision': 'training-corpus', 'questions': {'tools': {'option_words': 6}},
            'tools': [{'name': 'command', 'option': 'Run a command'}]}


def test_release_manifest_pins_all_heads_and_the_evaluated_vocabulary(tmp_path):
    result = release.build(corpus(), heads(tmp_path), 'test-release', 'revision')
    assert set(result['heads']) == {'turn-load', 'unit-rank', 'code-rank'}
    assert all(len(pin['sha256']) == 64 for pin in result['heads'].values())
    assert result['preload']['options'] == {'command': 'Run a command'}
    assert result['preload']['head_tokens'] == 512
    bound = corpus()
    bound['questions']['tools']['options'] = {'command': 'The evaluated text'}
    assert release.build(bound, tmp_path, 'test', 'revision')['preload']['options']['command'] == 'The evaluated text'


def test_release_manifest_rejects_a_different_corpus_or_encoding(tmp_path):
    heads(tmp_path)
    changed = corpus()
    changed['catalog_revision'] = 'another-corpus'
    with pytest.raises(ValueError, match='corpus differ'):
        release.build(changed, tmp_path, 'test', 'revision')
    changed = corpus()
    changed['questions']['tools']['independent'] = True
    with pytest.raises(ValueError, match='encodings differ'):
        release.build(changed, tmp_path, 'test', 'revision')
