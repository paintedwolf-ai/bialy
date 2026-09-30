"""detach.py STATE_DIR -- COMMAND...: run a long job outside the caller's session.

The job gets its own session and process group (so closing the terminal, an agent
restart, or a lost ssh connection does not signal it), stdin from /dev/null, output in
STATE_DIR/out.log, and its PID in STATE_DIR/pid; on macOS it runs under
`caffeinate -i` so the machine does not sleep mid-run. The job must be resumable
itself (bialy judge checkpoints every row); this only keeps it alive.
"""
import os
import shutil
import sys


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit("usage: detach.py STATE_DIR -- COMMAND...")
    state, command = sys.argv[1], sys.argv[3:]
    os.makedirs(state, exist_ok=True)
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    if os.fork():
        return
    os.setsid()
    log = os.open(os.path.join(state, "out.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    with open(os.path.join(state, "pid"), "w") as fh:
        fh.write(str(os.getpid()))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
