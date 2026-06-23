"""
Enhanced retrieve stage with query parsing and metadata filtering.
LoCoMo-specific version with global qa_idx support for multi-conversation files.

Usage:
    python retrieve_stage_enhanced_locomo.py --mode enhanced  # Use enhanced retrieval
    python retrieve_stage_enhanced_locomo.py --mode baseline  # Use simple retrieval
    python retrieve_stage_enhanced_locomo.py --mode both      # Run both for comparison
"""
import argparse
import os
import sys
import time
# Add parent directory to path to import memblock_extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import memblock_extractor as mx
from retrieval_impl_locomo import SimpleRetriever
from retrieval_enhanced_locomo import EnhancedRetriever as DefaultEnhancedRetriever
from retrieval_enhanced_locomo_multiretrieval import EnhancedRetriever as MultiRetrievalEnhancedRetriever

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
        "--final-content-file",
        type=str,
        default=None,
        help="Path to the finalized memory blocks (JSONL). If not set, uses default in output-dir."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to out/<run-id>)"
    )
    parser.add_argument(
        "--run-id", #/data/wjl/SA-Mem/out/locomo-user        /final_boxes_content_user_ids.jsonl
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
        "--output-suffix",
        type=str,
        default="",
        help="Appended to output filenames before the extension (e.g. '_test5' -> retrieval_enhanced_test5.csv). "
             "Also applied to retrieval_baseline.{csv,jsonl} and retrieve_stage_enhanced.log.",
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
        "--graph-hops",
        type=int,
        default=1,
        help="Number of SIMILAR-edge hops to expand (default: 1)."
    )
    parser.add_argument(
        "--graph-limit",
        type=int,
        default=200,
        help="Max graph rows to return per query."
    )
    parser.add_argument(
        "--graph-extract-source",
        choices=["event", "raw"],
        default=mx.Config.GRAPH_EXTRACT_SOURCE,
        help="Entity/relation extraction source for graph retrieval: event or raw content_text.",
    )
    parser.add_argument(
        "--graph-person-relations",
        action="store_true",
        help="Enable person-relation profile expansion in graph results.",
    )
    parser.add_argument(
        "--graph-no-relations",
        action="store_true",
        help="Disable entity relation triples when graph expansion is enabled."
    )
    parser.add_argument(
        "--graph-index-expand",
        action="store_true",
        help="Enable local graph-index retrieval. This is isolated from --graph-expand and does not run Neo4j graph expansion.",
    )
    parser.add_argument(
        "--graph-index-dir",
        type=str,
        default=None,
        help="Directory containing graph_user_index_<user_id>.json files. Defaults to <output-dir>/graph_user_exports.",
    )
    parser.add_argument(
        "--graph-index-boost",
        type=float,
        default=0.0,
        help="Deprecated/reserved. Local graph-index rerank currently uses fixed weights: semantic=0.70, entity=0.18, relation=0.12.",
    )
    parser.add_argument(
        "--use-anchor",
        action="store_true",
        default=False,
        help="Enable anchor-based temporal resolution for ANCHOR-type queries. "
             "When disabled, ANCHOR queries fall back to full pool (no anchor resolution). "
             "Default: False (anchor disabled)."
    )
    parser.add_argument(
        "--no-use-anchor",
        action="store_false",
        dest="use_anchor",
        help="Disable anchor-based temporal resolution for ANCHOR-type queries. "
             "Use this flag to force ANCHOR queries to fall back to full pool."
    )
    parser.add_argument(
        "--axis-mode",
        choices=["auto", "session", "event", "none"],
        default="auto",
        help="Bi-temporal ablation switch. "
             "auto: use QueryParser-inferred axis (default). "
             "session: force SESSION axis (only query session_tree); ANCHOR queries degrade to full pool. "
             "event: force EVENT axis (only query event_tree). "
             "none: skip the entire temporal filtering block."
    )
    args = parser.parse_args()
    start_ts = time.perf_counter()
    # Apply configuration
    mx.Config.apply_run_id(args.run_id)
    mx.Config.RAW_DATA_FILE = args.raw_data_file
    # # 兼容下游代码：如果下游使用 FINAL_CONTENT_FILE，则同步为传入的 raw-data-file
    # if getattr(args, "raw_data_file", None):
    #     mx.Config.FINAL_CONTENT_FILE = args.raw_data_file
        
    if args.final_content_file:
        mx.Config.FINAL_CONTENT_FILE = args.final_content_file

    mx.Config.GRAPH_EXTRACT_SOURCE = str(args.graph_extract_source or "event").strip().lower()
    
    mx.Config.LIMIT_CONVERSATIONS = None if args.limit_conversations == -1 else max(0, args.limit_conversations)

    if args.api_key is not None:
        mx.Config.API_KEY = args.api_key
    if args.base_url is not None:
        mx.Config.BASE_URL = args.base_url

    # 如果通过 CLI 指定了输出目录，则将其写入配置
    if args.output_dir:
        mx.Config.OUTPUT_DIR = args.output_dir

    # Apply --output-suffix to all produced filenames in this run.
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
    mx.logger.info(
        "ℹ️ Local graph-index expansion: %s",
        "enabled" if args.graph_index_expand else "disabled",
    )
    if args.graph_expand and args.graph_index_expand:
        mx.logger.warning("⚠️ --graph-index-expand is isolated from --graph-expand; Neo4j graph expansion will be disabled for enhanced retrieval.")

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
            graph_hops=args.graph_hops,
            graph_include_relations=not args.graph_no_relations,
            graph_person_relations=args.graph_person_relations,
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
        enhanced_cls = MultiRetrievalEnhancedRetriever if args.graph_index_expand else DefaultEnhancedRetriever
        enhanced_kwargs = {
            "top_k": mx.Config.TOP_K_RETRIEVE,
            "graph_expand": False if args.graph_index_expand else args.graph_expand,
            "graph_min_score": args.graph_min_score,
            "graph_limit": args.graph_limit,
            "graph_hops": args.graph_hops,
            "graph_include_relations": not args.graph_no_relations,
            "graph_person_relations": args.graph_person_relations,
            "use_anchor": args.use_anchor,
            "axis_mode": args.axis_mode,
        }
        if args.graph_index_expand:
            enhanced_kwargs.update({
                "graph_index_expand": True,
                "graph_index_dir": args.graph_index_dir,
                "graph_index_boost": args.graph_index_boost,
            })
        retr_enhanced = enhanced_cls(worker, **enhanced_kwargs)
        retr_enhanced.run(enhanced_jsonl, enhanced_csv, use_enhanced=True)
        mx.logger.info("⏱️ Enhanced retrieval time: %.2f seconds", time.perf_counter() - t1)

    mx.logger.info("✅ Retrieval complete!")
    elapsed = time.perf_counter() - start_ts
    mx.logger.info("⏱️ Total retrieval time: %.2f seconds", elapsed)
    if args.mode == "both":
        mx.logger.info("ℹ️ Comparison mode: Both baseline and enhanced results saved.")
        mx.logger.info("   Baseline: retrieval_baseline%s.jsonl, retrieval_baseline%s.csv", suffix, suffix)
        mx.logger.info("   Enhanced: retrieval_enhanced%s.jsonl, retrieval_enhanced%s.csv", suffix, suffix)


if __name__ == "__main__":
    main()
