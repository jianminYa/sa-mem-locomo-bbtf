#!/usr/bin/env python3
"""
Generic generation stage with configurable provider/model and graph-context controls.
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import memblock_extractor as mx
from generate_impl import AnswerGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate answers with enhanced controls")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=getattr(mx.Config, "LLM_PROVIDER", "openai"),
        help="LLM provider selector: openai (default) or ollama (local).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model name (applies to selected provider).",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="Override Ollama OpenAI-compatible base URL (default: http://localhost:11434/v1).",
    )
    parser.add_argument(
        "--ollama-api-key",
        type=str,
        default=None,
        help="Override Ollama API key (default: ollama).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override OpenAI-compatible base URL (or set OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override OpenAI-compatible API key (or set OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=None,
        help="Override Ollama context length (num_ctx).",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=None,
        help="Override Ollama max tokens to generate (num_predict).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for output directory")
    parser.add_argument("--retrieval-file", type=str, default=None,
                       help="Path to retrieval results JSONL file")
    parser.add_argument("--answer-topn", type=int, default=5,
                       help="Number of top blocks to use for answer generation")
    parser.add_argument("--text-modes", type=str, nargs="+", default=["content"],
                       help="Text modes for generation: content, event, content_trace_event, trace_event")
    parser.add_argument("--use-graph-context", action="store_true",
                       help="Append graph expansion context to the prompt (requires retrieval JSONL with graph field).")
    parser.add_argument(
        "--graph-context-events-only",
        action="store_true",
        help="Use only expanded/similar event context (skip relation triples).",
    )
    parser.add_argument(
        "--graph-context-relations-only",
        action="store_true",
        help="Use only relation triples (skip expanded/similar event context).",
    )
    parser.add_argument(
        "--graph-context-expanded-topk",
        type=int,
        default=None,
        help="Keep only top-k expanded graph events by query_similarity (fallback to graph_similarity_score) before prompt injection.",
    )
    parser.add_argument(
        "--graph-context-expanded-min-score",
        type=float,
        default=None,
        help="Minimum query_similarity (fallback to graph_similarity_score) for expanded graph events before prompt injection.",
    )
    parser.add_argument(
        "--graph-context-person-profile",
        action="store_true",
        help="Include person profile graph context built from person relation facts.",
    )
    parser.add_argument(
        "--graph-context-style",
        type=str,
        default="default",
        help="Graph context formatting style (e.g., default, fl-pro).",
    )
    parser.add_argument("--limit-conversations", type=int, default=None,
                       help="Limit number of conversations to process (must match build/retrieval)")
    parser.add_argument("--raw-data-file", type=str, default=None,
                       help="Path to raw data file (needed for loading QA pairs)")
    parser.add_argument("--output-suffix", type=str, default="",
                       help="Suffix for output files (e.g. 'top20')")

    args = parser.parse_args()

    # Apply run_id configuration
    mx.Config.apply_run_id(args.run_id)

    # LLM provider/model overrides
    mx.Config.LLM_PROVIDER = (args.provider or mx.Config.LLM_PROVIDER or "openai").strip().lower()
    if args.ollama_base_url is not None:
        mx.Config.OLLAMA_BASE_URL = args.ollama_base_url
    if args.ollama_api_key is not None:
        mx.Config.OLLAMA_API_KEY = args.ollama_api_key
    if args.base_url is not None:
        if mx.Config.LLM_PROVIDER == "ollama":
            mx.Config.OLLAMA_BASE_URL = args.base_url
        else:
            mx.Config.BASE_URL = args.base_url
    if args.api_key is not None:
        if mx.Config.LLM_PROVIDER == "ollama":
            mx.Config.OLLAMA_API_KEY = args.api_key
        else:
            mx.Config.API_KEY = args.api_key
    if args.llm_model:
        if mx.Config.LLM_PROVIDER == "ollama":
            mx.Config.OLLAMA_LLM_MODEL = args.llm_model
        else:
            mx.Config.LLM_MODEL = args.llm_model
    if args.ollama_num_ctx is not None:
        mx.Config.OLLAMA_NUM_CTX = int(args.ollama_num_ctx)
    if args.ollama_num_predict is not None:
        mx.Config.OLLAMA_NUM_PREDICT = int(args.ollama_num_predict)

    # Set limit conversations if provided
    if args.limit_conversations is not None:
        mx.Config.LIMIT_CONVERSATIONS = args.limit_conversations
        mx.logger.info("ℹ️ LIMIT_CONVERSATIONS set to %s", args.limit_conversations)

    # Set raw data file if provided
    if args.raw_data_file:
        mx.Config.RAW_DATA_FILE = args.raw_data_file

    # Set answer topn
    if args.answer_topn:
        mx.Config.ANSWER_TOP_N = args.answer_topn

    # Set text modes
    if args.text_modes:
        mx.Config.GEN_TEXT_MODES = args.text_modes

    # Determine retrieval file path
    if args.retrieval_file:
        retrieval_jsonl = args.retrieval_file
    else:
        default_enhanced = os.path.join(mx.Config.OUTPUT_DIR, "retrieval_enhanced.jsonl")
        retrieval_jsonl = default_enhanced if os.path.exists(default_enhanced) else mx.Config.SIMPLE_RETRIEVAL_JSONL

    mx.Config.USE_GRAPH_CONTEXT = bool(args.use_graph_context)
    # if args.graph_context_events_only and args.graph_context_relations_only:
    #     mx.logger.error("❌ Only one of --graph-context-events-only or --graph-context-relations-only can be set.")
    #     sys.exit(1)
    mx.Config.GRAPH_CONTEXT_EVENTS_ONLY = bool(args.graph_context_events_only)
    mx.Config.GRAPH_CONTEXT_RELATIONS_ONLY = bool(args.graph_context_relations_only)
    mx.Config.GRAPH_CONTEXT_EXPANDED_TOPK = (
        int(args.graph_context_expanded_topk)
        if args.graph_context_expanded_topk is not None and int(args.graph_context_expanded_topk) >= 0
        else None
    )
    mx.Config.GRAPH_CONTEXT_EXPANDED_MIN_SCORE = (
        float(args.graph_context_expanded_min_score)
        if args.graph_context_expanded_min_score is not None
        else None
    )
    mx.Config.GRAPH_CONTEXT_PERSON_PROFILE = bool(args.graph_context_person_profile)
    mx.Config.GRAPH_CONTEXT_STYLE = str(args.graph_context_style or "default").strip().lower()

    # Output files
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_jsonl = os.path.join(mx.Config.OUTPUT_DIR, f"generation_results{suffix}.jsonl")
    output_csv = os.path.join(mx.Config.OUTPUT_DIR, f"report_generation_qa{suffix}.csv")
    mx.Config.GEN_SUMMARY_FILE = os.path.join(
        mx.Config.OUTPUT_DIR,
        f"generation_metrics_summary{suffix}.jsonl",
    )

    mx.logger.info("=" * 60)
    mx.logger.info("🚀 Generation Stage")
    mx.logger.info("=" * 60)
    mx.logger.info("Run ID: %s", mx.Config.RUN_ID)
    mx.logger.info("Output directory: %s", mx.Config.OUTPUT_DIR)
    mx.logger.info("Retrieval file: %s", retrieval_jsonl)
    mx.logger.info("Answer TopN: %s", mx.Config.ANSWER_TOP_N)
    mx.logger.info("Text modes: %s", mx.Config.GEN_TEXT_MODES)
    mx.logger.info("Graph expanded topk: %s", getattr(mx.Config, "GRAPH_CONTEXT_EXPANDED_TOPK", None))
    mx.logger.info("Graph expanded min_score: %s", getattr(mx.Config, "GRAPH_CONTEXT_EXPANDED_MIN_SCORE", None))
    mx.logger.info("Limit conversations: %s", mx.Config.LIMIT_CONVERSATIONS)
    mx.logger.info("LLM provider: %s", mx.Config.LLM_PROVIDER)
    mx.logger.info("LLM model: %s", mx.Config.effective_llm_model())
    mx.logger.info("LLM base_url: %s", mx.Config.effective_base_url())
    if mx.Config.LLM_PROVIDER == "ollama":
        mx.logger.info("Ollama num_ctx: %s", getattr(mx.Config, "OLLAMA_NUM_CTX", None))
        mx.logger.info("Ollama num_predict: %s", getattr(mx.Config, "OLLAMA_NUM_PREDICT", None))
    mx.logger.info("=" * 60)

    if not os.path.exists(retrieval_jsonl):
        mx.logger.error("❌ Retrieval file not found: %s", retrieval_jsonl)
        mx.logger.error("Please run retrieval stage first!")
        sys.exit(1)

    # Initialize worker (reads from Config)
    worker = mx.LLMWorker()

    # Initialize generator
    generator = AnswerGenerator(
        worker=worker,
        answer_topn=mx.Config.ANSWER_TOP_N,
        text_modes=mx.Config.GEN_TEXT_MODES,
        stage_label="gen",
    )

    # Run generation
    mx.logger.info("🔄 Starting generation...")
    generator.run(
        retrieval_jsonl=retrieval_jsonl,
        base_out_jsonl=output_jsonl,
        base_out_csv=output_csv,
    )

    mx.logger.info("=" * 60)
    mx.logger.info("✅ Generation Complete!")
    mx.logger.info("📄 Results saved to:")
    mx.logger.info("   - %s", output_jsonl)
    mx.logger.info("   - %s", output_csv)
    mx.logger.info("=" * 60)


if __name__ == "__main__":
    main()