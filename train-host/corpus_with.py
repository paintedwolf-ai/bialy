"""corpus_with.py IN OUT key=value...: a corpus copy with decision thresholds replaced, e.g.
tools.load_at=0.36 guides.omit_below=0.18 request.max_loads=3 skills.preload_at=3.32."""
import json
import sys

data = json.load(open(sys.argv[1]))
for arg in sys.argv[3:]:
    path, value = arg.split("=", 1)
    section, key = path.split(".")
    target = data["request"] if section == "request" else data["questions"][section]
    target[key] = type(target[key])(float(value)) if key in target else float(value)
json.dump(data, open(sys.argv[2], "w"))
