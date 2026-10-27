import sqlite3
import subprocess
from pathlib import Path


def test_retained_backup_opens_without_wal_side_files(tmp_path):
    source, output = tmp_path / 'source.db', tmp_path / 'capture'
    output.mkdir()
    script = (Path(__file__).parents[1] / 'runner/run-shard.sh').read_text()
    start = script.index('sqlite3 /cfg/store.db ')
    end = script.index('\nif [', start)
    export = script[start:end].replace('/cfg/store.db', '"$source"')
    with sqlite3.connect(source) as database:
        database.execute('PRAGMA journal_mode=WAL')
        database.execute('CREATE TABLE evidence(value TEXT)')
        database.execute("INSERT INTO evidence VALUES('committed')")
        database.commit()
        subprocess.run(['bash', '-euc', export], env={'PATH': '/usr/bin:/bin:/usr/local/bin',
                       'source': str(source), 'out': str(output)}, check=True)
        assert database.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    retained = output / 'store.db'
    assert retained.read_bytes()[18:20] == b'\x01\x01'
    for suffix in ('-wal', '-shm'):
        retained.with_name(retained.name + suffix).unlink(missing_ok=True)
    retained.chmod(0o400)
    output.chmod(0o500)
    try:
        with sqlite3.connect('file:' + str(retained) + '?mode=ro', uri=True) as database:
            assert database.execute('SELECT value FROM evidence').fetchone()[0] == 'committed'
        assert [p.name for p in output.iterdir()] == ['store.db']
    finally:
        output.chmod(0o700)
