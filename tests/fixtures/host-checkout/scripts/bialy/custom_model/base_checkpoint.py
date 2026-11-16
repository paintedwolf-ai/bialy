"""The arguments the host's base_checkpoint.py takes."""
import argparse

parser = argparse.ArgumentParser()
for name in ("model", "out"):
    parser.add_argument("--" + name, required=True)
