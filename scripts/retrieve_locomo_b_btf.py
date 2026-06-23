#!/usr/bin/env python3
"""Run LoCoMo retrieval with an explicit top-k.

The README entrypoint `retrieval/retrieve_stage_enhanced_locomo.py` reads
`memblock_extractor.Config.TOP_K_RETRIEVE` and does not expose a CLI flag for it.
This wrapper sets the value before executing that entrypoint.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--top-k", type=int, default=5)
    known, rest = parser.parse_known_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    retrieval_dir = os.path.join(repo_root, "retrieval")
    sys.path.insert(0, repo_root)
    sys.path.insert(0, retrieval_dir)

    import memblock_extractor as mx

    mx.Config.TOP_K_RETRIEVE = max(1, int(known.top_k))

    target = os.path.join(retrieval_dir, "retrieve_stage_enhanced_locomo.py")
    sys.argv = [target, *rest]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
