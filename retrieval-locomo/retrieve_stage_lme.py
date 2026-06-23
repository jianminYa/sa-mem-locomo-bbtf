"""
LongMemEval retrieval stage entry point.

Usage examples
--------------
# Enhanced retrieval (default), use gpt + gpt2 boxes:
python retrieval/retrieve_stage_lme.py \\
    --boxes-dirs /data/lyc/SA-Mem/out/longmemeval_s_gpt \\
                 /data/lyc/SA-Mem/out/longmemeval_s_gpt2 \\
    --lme-data-dir /data/wjl/SA-Mem/data/lme_preprocessed \\
    --output-dir /data/lyc/SA-Mem/out/longmemeval_s_lme \\
    --mode enhanced --overwrite

# Baseline retrieval:
python retrieval/retrieve_stage_lme.py \\
    --boxes-dirs /data/lyc/SA-Mem/out/longmemeval_s_gpt \\
                 /data/lyc/SA-Mem/out/longmemeval_s_gpt2 \\
    --lme-data-dir /data/wjl/SA-Mem/data/lme_preprocessed \\
    --output-dir /data/lyc/SA-Mem/out/longmemeval_s_lme \\
    --mode baseline --overwrite

# Both modes at once:
python retrieval/retrieve_stage_lme.py \\
    --boxes-dirs /data/lyc/SA-Mem/out/longmemeval_s_gpt \\
                 /data/lyc/SA-Mem/out/longmemeval_s_gpt2 \\
    --lme-data-dir /data/wjl/SA-Mem/data/lme_preprocessed \\
    --output-dir /data/lyc/SA-Mem/out/longmemeval_s_lme \\
    --mode both --overwrite

Notes
-----
- boxes-dirs order matters: later dirs overwrite earlier ones for duplicate
  user_ids (e.g., 1192316e appears in both gpt and gpt2; gpt2 is listed last
  so gpt2's blocks are used).
- gpt3 boxes are intentionally excluded because their user_id is all "locomo10"
  (a known bug), making them unusable for LME retrieval.
- The script reads lme_preprocessed/*.json files (8-char hex names only, no
  _abs variants) and skips questions without corresponding blocks.
"""
import argparse
import os
import sys
import time

# Ensure project root is on path before local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import memblock_extractor as mx
from retrieval_lme import LMERetriever

# Default paths
_DEFAULT_BOXES_DIRS = [
    "/data/lyc/SA-Mem/out/longmemeval_s_gpt",
    "/data/lyc/SA-Mem/out/longmemeval_s_gpt2",
]
_DEFAULT_LME_DATA_DIR = "/data/wjl/SA-Mem/data/lme_preprocessed"
_DEFAULT_OUTPUT_DIR = "/data/lyc/SA-Mem/out/longmemeval_s_lme"


def _maybe_remove(path: str, overwrite: bool) -> None:
    if not overwrite:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            mx.logger.info("🗑️ Removed existing file: %s", path)
    except Exception as e:
        mx.logger.warning("⚠️ Failed to remove %s: %s", path, e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval memory retrieval stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "enhanced", "both"],
        default="enhanced",
        help="Retrieval mode (default: enhanced)",
    )
    parser.add_argument(
        "--boxes-dirs",
        nargs="+",
        default=_DEFAULT_BOXES_DIRS,
        metavar="DIR",
        help=(
            "One or more output directories containing final_boxes_content.jsonl "
            "and vector_store/. Dirs are merged in order; later dirs win on "
            "duplicate user_ids. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--lme-data-dir",
        default=_DEFAULT_LME_DATA_DIR,
        metavar="DIR",
        help="Directory of lme_preprocessed JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Output directory for retrieval results (default: %(default)s)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional run identifier. When set, outputs go to "
            "<output-dir>/<run-id>/ instead of <output-dir>/"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output files before writing (prevents duplicates)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N questions (useful for quick debugging)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override TOP_K_RETRIEVE (default: use Config value = 20)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override OPENAI_API_KEY",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override OPENAI_BASE_URL",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help=(
            "Appended to output filenames before the extension "
            "(e.g. '_test5' → retrieval_enhanced_test5.jsonl)"
        ),
    )
    parser.add_argument(
        "--graph-expand",
        action="store_true",
        help="Expand results with graph events/relations (requires Memgraph).",
    )
    parser.add_argument(
        "--graph-min-score",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--graph-limit",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--graph-no-relations",
        action="store_true",
        help="Disable entity relation triples in graph expansion.",
    )
    parser.add_argument(
        "--use-anchor",
        action="store_true",
        default=False,
        help="Enable anchor-based temporal resolution. Default: False (anchor disabled).",
    )
    parser.add_argument(
        "--no-use-anchor",
        action="store_false",
        dest="use_anchor",
        help="Disable anchor resolution (default behavior).",
    )
    parser.add_argument(
        "--axis-mode",
        choices=["auto", "session", "event", "none"],
        default="auto",
        help="Bi-temporal ablation switch. "
             "auto: use QueryParser-inferred axis. "
             "session: force SESSION (only session_tree); ANCHOR degrades to full pool. "
             "event: force EVENT (only event_tree). "
             "none: skip the entire temporal filtering block.",
    )

    args = parser.parse_args()
    start_ts = time.perf_counter()

    # ---- Configure paths ----
    output_dir = args.output_dir
    if args.run_id:
        output_dir = os.path.join(output_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Point Config to the output dir so EmbeddingStore / TraceLogger
    # use the right base paths.
    mx.Config.OUTPUT_DIR = output_dir
    # VECTOR_DIR will be overridden per-user inside LMERetriever._score_and_rank
    mx.Config.VECTOR_DIR = os.path.join(output_dir, "vector_store")
    os.makedirs(mx.Config.VECTOR_DIR, exist_ok=True)

    if args.top_k is not None:
        mx.Config.TOP_K_RETRIEVE = args.top_k
    if args.api_key is not None:
        mx.Config.API_KEY = args.api_key
    if args.base_url is not None:
        mx.Config.BASE_URL = args.base_url

    suffix = (args.output_suffix or "").strip()

    # ---- Setup logging ----
    log_file = os.path.join(output_dir, f"retrieve_stage_lme{suffix}.log")
    mx.setup_logging(log_file)

    mx.logger.info("=" * 60)
    mx.logger.info("LongMemEval Retrieval Stage")
    mx.logger.info("  mode        : %s", args.mode)
    mx.logger.info("  boxes_dirs  : %s", args.boxes_dirs)
    mx.logger.info("  lme_data_dir: %s", args.lme_data_dir)
    mx.logger.info("  output_dir  : %s", output_dir)
    mx.logger.info("  top_k       : %s", mx.Config.TOP_K_RETRIEVE)
    mx.logger.info("  limit       : %s", args.limit)
    mx.logger.info("  overwrite   : %s", args.overwrite)
    mx.logger.info("  use_anchor  : %s", args.use_anchor)
    mx.logger.info("  axis_mode   : %s", args.axis_mode)
    mx.logger.info("=" * 60)

    if not (mx.Config.API_KEY or "").strip():
        mx.logger.warning("⚠️  OPENAI_API_KEY is missing; LLM calls will fail.")

    # Validate boxes dirs
    for d in args.boxes_dirs:
        jsonl = os.path.join(d, "final_boxes_content.jsonl")
        if not os.path.exists(jsonl):
            mx.logger.warning("⚠️ boxes file not found (will be skipped): %s", jsonl)

    # Validate lme_data_dir
    if not os.path.isdir(args.lme_data_dir):
        mx.logger.error("❌ lme_data_dir does not exist: %s", args.lme_data_dir)
        sys.exit(1)

    # Build (run_id, out_dir) pairs for LMERetriever.
    # run_id derived from basename of the directory.
    boxes_dirs_pairs = [
        (os.path.basename(d), d) for d in args.boxes_dirs
    ]

    worker = mx.LLMWorker()

    def _build_retriever():
        return LMERetriever(
            worker=worker,
            boxes_dirs=boxes_dirs_pairs,
            lme_data_dir=args.lme_data_dir,
            top_k=mx.Config.TOP_K_RETRIEVE,
            graph_expand=args.graph_expand,
            graph_min_score=args.graph_min_score,
            graph_limit=args.graph_limit,
            graph_include_relations=not args.graph_no_relations,
            use_anchor=args.use_anchor,
            axis_mode=args.axis_mode,
        )

    # ---- Baseline ----
    if args.mode in ("baseline", "both"):
        mx.logger.info("🔍 Running BASELINE retrieval...")
        bl_jsonl = os.path.join(output_dir, f"retrieval_baseline{suffix}.jsonl")
        bl_csv = os.path.join(output_dir, f"retrieval_baseline{suffix}.csv")
        _maybe_remove(bl_jsonl, args.overwrite)
        _maybe_remove(bl_csv, args.overwrite)
        t0 = time.perf_counter()
        retr_bl = _build_retriever()
        retr_bl.run(bl_jsonl, bl_csv, use_enhanced=False, limit=args.limit)
        mx.logger.info("⏱️ Baseline time: %.2f s", time.perf_counter() - t0)

    # ---- Enhanced ----
    if args.mode in ("enhanced", "both"):
        mx.logger.info("🔍 Running ENHANCED retrieval...")
        en_jsonl = os.path.join(output_dir, f"retrieval_enhanced{suffix}.jsonl")
        en_csv = os.path.join(output_dir, f"retrieval_enhanced{suffix}.csv")
        _maybe_remove(en_jsonl, args.overwrite)
        _maybe_remove(en_csv, args.overwrite)
        t1 = time.perf_counter()
        retr_en = _build_retriever()
        retr_en.run(en_jsonl, en_csv, use_enhanced=True, limit=args.limit)
        mx.logger.info("⏱️ Enhanced time: %.2f s", time.perf_counter() - t1)

    elapsed = time.perf_counter() - start_ts
    mx.logger.info("✅ Total elapsed: %.2f s", elapsed)
    mx.logger.info("📁 Results in: %s", output_dir)


if __name__ == "__main__":
    main()
