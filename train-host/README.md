# Training host

What the factory needs on the machine that trains.

| File | Does |
|---|---|
| `requirements.txt` | The trainer's pinned packages. `bialy run` and a campaign's selector make the trainer environment from it with `uv`, under `<root>/venvs/train`, with the torch wheel index chosen from the accelerator found. `laya` is pinned here and nowhere else; the pin is the version whose API the pinned checkout's `scripts/bialy/train.py` uses. |
| `student/` | The Gemma 4 student hosts. |

Heads are trained, calibrated, qualified and released by the selector
(`src/bialy/campaign/selector.py`), through `bialy run` or a campaign. It calls
the pinned Painted Wolf Code checkout's `scripts/bialy/` trainers, calibration,
replay and parity probe; nothing here trains a head by hand. See
[`docs/running.md`](../docs/running.md).
