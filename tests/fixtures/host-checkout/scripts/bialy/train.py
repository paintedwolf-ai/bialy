import argparse
import os

MODEL_ID = os.environ.get("LYCAON_DECIDE_MODEL_ID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--train")
    ap.add_argument("--val")
    ap.add_argument("--out")
    ap.add_argument("--model")
    ap.add_argument("--epochs")
    ap.add_argument("--patience")
    ap.add_argument("--batch-size")
    ap.add_argument("--lr")
    ap.add_argument("--seed")
    ap.add_argument("--label")
    ap.add_argument("--pos-weight")
    ap.add_argument("--families")
    ap.add_argument("--skill-scored")
    ap.add_argument("--skill-zeros")
    ap.add_argument("--turn-rows")
    ap.add_argument("--tool-truth")
    ap.add_argument("--rank-levels")
    ap.add_argument("--tool-weight")
    ap.add_argument("--tool-negatives")
    ap.add_argument("--retention-truth")
    return ap.parse_args()
