import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--examples")
    ap.add_argument("--engine")
    ap.add_argument("--tool-beta")
    ap.add_argument("--request-beta")
    ap.add_argument("--tool-truth")
    ap.add_argument("--preload-precision")
    return ap.parse_args()
