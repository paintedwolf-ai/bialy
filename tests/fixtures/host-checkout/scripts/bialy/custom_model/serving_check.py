"""The arguments the host's serving_check.py takes."""
import argparse

parser = argparse.ArgumentParser()
for name in ("out",):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--url")
parser.add_argument("--revision")
