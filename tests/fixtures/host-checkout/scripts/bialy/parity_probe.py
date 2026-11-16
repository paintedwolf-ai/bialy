import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--head")
    ap.add_argument("--examples")
    ap.add_argument("--engine")
    ap.add_argument("--corpus")
    ap.add_argument("--n")
    ap.add_argument("--tolerance")
    return ap.parse_args()
