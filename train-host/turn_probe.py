"""Score every option of a turn head with the trainer's own forward, tools and guides alike.

Writes one row per complete example: the row's key, offered candidates and labels, and
the head's probability for every tool and guide option, encoded exactly as the engine
encodes them (one option per row for an independent head, the roster for a joint one).
The output feeds `guide_audit.py` and `turn_score.py` without an engine build.

Usage: turn_probe.py --trainer DIR --corpus FILE --rows FILE --head FILE --out FILE [--limit N]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def option_probs(agent, model, corpus, ex, device, kind, qid, family, torch, multi_item, precompute, forward_head, state_text):
    """Probabilities per option of one multi question, encoded as the engine would."""
    context = int(agent.cfg.get("max_len", 512))
    head_max_len = min(int(corpus.state_spec.get("head_tokens", 512)), context - 64)
    question = corpus.multi_question(kind, ex["offered"]["loadable" if kind == "tool" else "guides"])
    if not question["options"]:
        return {}
    state = state_text(ex["state"])
    if question.get("independent"):
        items, names = [], []
        for name, text in question["options"].items():
            one = dict(question, options={name: text})
            items.append(multi_item(agent.tok, state, one, {name: 0}, context, head_max_len, family))
            names.append(name)
    else:
        item = multi_item(agent.tok, state, question, {k: 0 for k in question["options"]}, context, head_max_len, family)
        items, names = [item], list(question["options"].keys())[: len(item["markers"])]
    features = precompute(agent, items, device, batch_size=64)
    out = {}
    with torch.no_grad():
        for f, item in zip(features, items):
            logits = forward_head(model, f["h"][None].to(device), f["att"][None].to(device), f["marker_pos"][None].to(device),
                                  f["marker_mask"][None].to(device), torch.tensor([item["qtype"]], device=device))[0].cpu()
            probs = torch.sigmoid(logits[: len(item["markers"])]).tolist()
            if question.get("independent"):
                out[names[len(out)]] = probs[0]
            else:
                out.update(zip(names, probs))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trainer", "corpus", "rows", "head", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--limit", type=int, default=0, help="complete rows to score; 0 scores every complete row")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.trainer) / "scripts/decide"))
    import laya
    import torch
    from corpus import Corpus
    from headfile import read_metadata
    from rows import load
    from safetensors.torch import load_file
    from train import forward_head, multi_item, precompute, state_text

    torch.set_num_threads(8)
    device = os.environ.get("LYCAON_DECIDE_DEVICE", "cpu")
    corpus = Corpus.load(args.corpus)
    agent = laya.load(args.model, device=device)
    meta = read_metadata(args.head)[0]
    context = int(agent.cfg.get("max_len", 512))
    budget = min(int(corpus.state_spec.get("head_tokens", 512)), context - 64)
    encoding = "independent" if corpus.independent() else "joint"
    if (meta.get("max_len"), meta.get("head_max_len"), meta.get("tool_encoding", "joint")) != (str(context), str(budget), encoding):
        raise ValueError("head encoding metadata does not match inference: %s" % {k: meta.get(k) for k in ("max_len", "head_max_len", "tool_encoding")})
    tensors = load_file(args.head, device=device)
    for name in ("head", "scorer", "type_emb"):
        getattr(agent.model, name).load_state_dict({k[len(name) + 1:]: v for k, v in tensors.items() if k.startswith(name + ".")})
    agent.model.eval()
    examples = [r for r in load(args.rows) if not r["partial"]]
    if args.limit:
        examples = examples[: args.limit]
    families = set((meta.get("families") or "tools,guides").split(","))
    signature = {name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest() for name in ("head", "corpus", "rows")}
    helpers = (torch, multi_item, precompute, forward_head, state_text)
    with Path(args.out).open("x") as out:
        for n, row in enumerate(examples, 1):
            answers = {}
            if "tools" in families:
                answers["tools"] = {"probabilities": option_probs(agent, agent.model, corpus, row, device, "tool", "tools", "tools", *helpers)}
            if "guides" in families:
                answers["guides"] = {"probabilities": option_probs(agent, agent.model, corpus, row, device, "guide", "guides", "guides", *helpers)}
            json.dump({"key": "%s:%s" % (row["session"], row["receipt"]), "host": row["host"], "offered": row["offered"],
                       "labels": {"tools": row["labels"].get("tools", []), "guides": row["labels"].get("guides", {})},
                       "answers": answers}, out)
            out.write("\n")
            if n % 100 == 0:
                out.flush()
                sys.stderr.write("scored %d/%d\n" % (n, len(examples)))
    Path(args.out + ".meta.json").write_text(json.dumps({"sha256": signature, "encoding": meta, "device": device, "rows": len(examples)}, indent=2))


if __name__ == "__main__":
    main()
