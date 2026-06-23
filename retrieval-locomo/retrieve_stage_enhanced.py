"""
Enhanced retrieve stage with query parsing and metadata filtering.
Usage:
    python retrieve_stage_enhanced.py --mode enhanced  # Use enhanced retrieval
    python retrieve_stage_enhanced.py --mode baseline  # Use simple retrieval
    python retrieve_stage_enhanced.py --mode both      # Run both for comparison
"""
import argparse
import os
import sys
import time
# Add parent directory to path to import memblock_extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import memblock_extractor as mx
from retrieval_impl import SimpleRetriever
from retrieval_enhanced import EnhancedRetriever

def _maybe_remove(path: str, overwrite: bool):
    if not overwrite:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        mx.logger.warning("⚠️ Failed to remove %s: %s", path, e)

def main():
    parser = argparse.ArgumentParser(description="Memory retrieval with optional query parsing")
    parser.add_argument(
        "--mode",
        choices=["baseline", "enhanced", "both"],
        default="enhanced",
        help="Retrieval mode: baseline (simple), enhanced (with query parsing), or both"
    )
    parser.add_argument(
        "--raw-data-file",
        type=str,
        default=mx.Config.RAW_DATA_FILE,
        help="Raw data path (JSON list or JSONL)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=mx.Config.OUTPUT_BASE_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run identifier (defaults to model name)"
    )
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=mx.Config.LIMIT_CONVERSATIONS if mx.Config.LIMIT_CONVERSATIONS is not None else -1,
        help="Limit number of conversations to process (-1 means no limit)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override API key (or set OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override base URL (or set OPENAI_BASE_URL)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove existing output files before writing (prevents appending duplicates)."
    )
    parser.add_argument(
        "--graph-expand",
        action="store_true",
        help="If set, expand results with Neo4j events/similar edges/relations."
    )
    parser.add_argument(
        "--graph-min-score",
        type=float,
        default=0.7,
        help="Similarity threshold for 1-hop graph expansion."
    )
    parser.add_argument(
        "--graph-limit",
        type=int,
        default=200,
        help="Max graph rows to return per query."
    )
    parser.add_argument(
        "--graph-no-relations",
        action="store_true",
        help="Disable entity relation triples when graph expansion is enabled."
    )
    parser.add_argument(
        "--use-anchor",
        action="store_true",
        default=False,
        help="Enable anchor-based temporal resolution. Default: False (anchor disabled)."
    )
    parser.add_argument(
        "--no-use-anchor",
        action="store_false",
        dest="use_anchor",
        help="Disable anchor resolution (default behavior)."
    )
    parser.add_argument(
        "--axis-mode",
        choices=["auto", "session", "event", "none"],
        default="auto",
        help="Bi-temporal ablation switch. "
             "auto: use QueryParser-inferred axis. "
             "session: force SESSION (only session_tree); ANCHOR degrades to full pool. "
             "event: force EVENT (only event_tree). "
             "none: skip the entire temporal filtering block."
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to output filenames before extension (e.g. '_axis_session')."
    )
    args = parser.parse_args()
    start_ts = time.perf_counter()
    # Apply configuration
    mx.Config.apply_run_id(args.run_id)
    mx.Config.RAW_DATA_FILE = args.raw_data_file
    mx.Config.LIMIT_CONVERSATIONS = None if args.limit_conversations == -1 else max(0, args.limit_conversations)

    if args.api_key is not None:
        mx.Config.API_KEY = args.api_key
    if args.base_url is not None:
        mx.Config.BASE_URL = args.base_url

    suffix = (args.output_suffix or "").strip()

    # Setup logging with file output
    log_file = os.path.join(mx.Config.OUTPUT_DIR, f"retrieve_stage_enhanced{suffix}.log")
    mx.setup_logging(log_file)

    if not (mx.Config.API_KEY or "").strip():
        mx.logger.warning("⚠️  OPENAI_API_KEY missing; LLM calls may fail.")

    worker = mx.LLMWorker()

    mx.logger.info("ℹ️ Using run_id=%s, output_dir=%s", mx.Config.RUN_ID, mx.Config.OUTPUT_DIR)
    mx.logger.info("ℹ️ Log file: %s", log_file)
    mx.logger.info("ℹ️ Retrieval mode: %s", args.mode)
    mx.logger.info("ℹ️ Anchor resolution: %s", "enabled" if args.use_anchor else "disabled")
    mx.logger.info("ℹ️ Axis mode: %s", args.axis_mode)

    # Run retrieval based on mode
    if args.mode in ("baseline", "both"):
        mx.logger.info("🔍 Running BASELINE retrieval (simple vector similarity)...")
        baseline_jsonl = os.path.join(mx.Config.OUTPUT_DIR, f"retrieval_baseline{suffix}.jsonl")
        baseline_csv = os.path.join(mx.Config.OUTPUT_DIR, f"retrieval_baseline{suffix}.csv")
        _maybe_remove(baseline_jsonl, args.overwrite)
        _maybe_remove(baseline_csv, args.overwrite)
        t0 = time.perf_counter()
        retr_baseline = SimpleRetriever(
            worker,
            top_k=mx.Config.TOP_K_RETRIEVE,
            graph_expand=args.graph_expand,
            graph_min_score=args.graph_min_score,
            graph_limit=args.graph_limit,
            graph_include_relations=not args.graph_no_relations,
        )
        retr_baseline.run(baseline_jsonl, baseline_csv)
        mx.logger.info("⏱️ Baseline retrieval time: %.2f seconds", time.perf_counter() - t0)
    if args.mode in ("enhanced", "both"):
        mx.logger.info("🔍 Running ENHANCED retrieval (with query parsing and metadata filtering)...")
        enhanced_jsonl = os.path.join(mx.Config.OUTPUT_DIR, f"retrieval_enhanced{suffix}.jsonl")
        enhanced_csv = os.path.join(mx.Config.OUTPUT_DIR, f"retrieval_enhanced{suffix}.csv")
        _maybe_remove(enhanced_jsonl, args.overwrite)
        _maybe_remove(enhanced_csv, args.overwrite)
        t1 = time.perf_counter()
        retr_enhanced = EnhancedRetriever(
            worker,
            top_k=mx.Config.TOP_K_RETRIEVE,
            graph_expand=args.graph_expand,
            graph_min_score=args.graph_min_score,
            graph_limit=args.graph_limit,
            graph_include_relations=not args.graph_no_relations,
            use_anchor=args.use_anchor,
            axis_mode=args.axis_mode,
        )
        retr_enhanced.run(enhanced_jsonl, enhanced_csv, use_enhanced=True)
        mx.logger.info("⏱️ Enhanced retrieval time: %.2f seconds", time.perf_counter() - t1)

    mx.logger.info("✅ Retrieval complete!")
    elapsed = time.perf_counter() - start_ts
    mx.logger.info("⏱️ Total retrieval time: %.2f seconds", elapsed)
    if args.mode == "both":
        mx.logger.info("ℹ️ Comparison mode: Both baseline and enhanced results saved.")
        mx.logger.info("   Baseline: retrieval_baseline.jsonl, retrieval_baseline.csv")
        mx.logger.info("   Enhanced: retrieval_enhanced.jsonl, retrieval_enhanced.csv")


if __name__ == "__main__":
    main()
