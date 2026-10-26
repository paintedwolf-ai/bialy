"""Select complete, already admitted examples for an automation rehearsal."""
import argparse
import json
import sys
from pathlib import Path


def select(rows, encode, limits, count):
    measured = [(len(encode(row)["input_ids"]), index, row) for index, row in enumerate(rows)]
    admitted = sorted((size, index, row) for size, index, row in measured if size <= limits)
    if len(admitted) < count:
        raise ValueError("insufficient complete examples within the dry-run token limit")
    chosen = admitted[:count]
    return [row for _, _, row in chosen], {
        "source_rows": len(rows), "selected_rows": count, "oversize_rows": sum(size > limits for size, _, _ in measured),
        "selected_tokens": [size for size, _, _ in chosen], "source_indices": [index for _, index, _ in chosen],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "model", "splits", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout / "scripts/bialy/custom_model"))
    from data import encode, load_split
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {}
    for name, partition, count in (("train.jsonl", "train", 2), ("val.jsonl", "validation", 1),
                                    ("holdout.jsonl", "system_test", 1)):
        rows = load_split(args.splits / name, partition)
        chosen, report[name] = select(rows, lambda row: encode(tokenizer, row, row["target"], 65536),
                                       args.max_tokens, count)
        (args.out / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen))
    (args.out / "selection.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
