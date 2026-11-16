"""The arguments the host's gguf_export.py takes."""
import argparse

parser = argparse.ArgumentParser()
for name in ("merge", "llama-cpp", "quantizer", "server", "out"):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--revision")
parser.add_argument("--resume")
