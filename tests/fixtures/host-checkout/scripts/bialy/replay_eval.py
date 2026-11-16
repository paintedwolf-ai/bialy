import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--examples")
    ap.add_argument("--engine")
    ap.add_argument("--limit")
    ap.add_argument("--tool-truth")
    ap.add_argument("--preload-at")
    ap.add_argument("--json")
    return ap.parse_args()
