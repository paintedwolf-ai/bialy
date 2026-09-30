import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from bialy.usage import record_usage


def test_usage_ledger_is_opt_in_and_keeps_concurrent_records_without_content(tmp_path, monkeypatch):
    path = tmp_path / 'usage.jsonl'
    reply = SimpleNamespace(id='request', usage=SimpleNamespace(model_dump=lambda: {'prompt_tokens': 12, 'completion_tokens': 20}),
                            choices=[SimpleNamespace(finish_reason='stop', message='private response')])
    monkeypatch.delenv('BIALY_USAGE_JSONL', raising=False)
    record_usage('model', reply)
    assert not path.exists()
    monkeypatch.setenv('BIALY_USAGE_JSONL', str(path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: record_usage('model', reply), range(30)))
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(entries) == 30
    assert all(e['usage']['completion_tokens'] == 20 for e in entries)
    assert 'private response' not in path.read_text()
