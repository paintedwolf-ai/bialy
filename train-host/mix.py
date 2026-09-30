"""mix.py OUT FRACTION SEED ROWS...: concatenate row files, keeping a seeded fraction of the
generated skill-request families (meta.source skillreq) and every session row."""
import hashlib
import json
import sys

out, fraction, seed = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open(out, "w", encoding="utf-8") as fh:
    for path in sys.argv[4:]:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            meta = row.get("meta") or {}
            if meta.get("source") == "skillreq":
                if meta.get("split") != "train":
                    continue
                if int(hashlib.sha256((seed + meta["family"]).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF >= fraction:
                    continue
            fh.write(line if line.endswith("\n") else line + "\n")
