"""Worker liveness and terminal outcomes independent of SSH attachment."""
import json
import threading
import time
from pathlib import Path

from . import jobs
from .resources import write


def train(directory):
    directory = Path(directory)
    stopped = threading.Event()
    def pulse():
        while not stopped.is_set():
            write(directory / 'heartbeat.json', {'at': time.time()})
            stopped.wait(10)
    heartbeat = threading.Thread(target=pulse, daemon=True)
    heartbeat.start()
    try:
        return jobs.train(directory)
    except jobs.JobBusy:
        raise
    except Exception as exc:
        path = directory / 'execution.json'
        state = json.loads(path.read_text()) if path.exists() else {'attempts': 0}
        write(path, dict(state, status='failed', failure={'type': type(exc).__name__, 'diagnostic': str(exc)}))
        raise
    finally:
        stopped.set()
        heartbeat.join()
