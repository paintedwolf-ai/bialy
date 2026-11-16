import argparse

import partition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--out")
    ap.add_argument("--model")
    ap.add_argument("--label")
    ap.add_argument("--epochs")
    ap.add_argument("--batch-size")
    ap.add_argument("--lr")
    ap.add_argument("--val-fraction")
    ap.add_argument("--max-rows")
    ap.add_argument("--seed")
    partition.arguments(ap)
    return ap.parse_args()
