"""Optional provider usage ledger, excluding prompts, responses, and credentials."""
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def record_usage(model, reply):
    target = os.environ.get('BIALY_USAGE_JSONL')
    if not target:
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {'time': datetime.now(timezone.utc).isoformat(), 'model': model,
             'request_id': reply.id, 'usage': reply.usage.model_dump() if reply.usage else {},
             'finish_reason': reply.choices[0].finish_reason}
    with path.open('a', encoding='utf-8') as out:
        fcntl.flock(out, fcntl.LOCK_EX)
        out.write(json.dumps(entry) + '\n')
        out.flush()
