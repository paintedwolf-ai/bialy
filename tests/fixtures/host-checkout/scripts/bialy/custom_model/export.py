"""The arguments the host's export.py takes."""
import argparse

parser = argparse.ArgumentParser()
for name in ("model", "run", "out"):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--resume")
