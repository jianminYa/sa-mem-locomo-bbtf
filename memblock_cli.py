import argparse
import json
import os
import pathlib
import queue
import re
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from typing import Any, Dict, List, Tuple

from memblock_extractor import (
    AnswerGenerator,
    Config,
    LLMWorker,
    MemoryBuilder,
    SimpleRetriever,
    TokenAnalyzer,
    TraceLinker,
    _announce_outputs,  # if you removed it from extractor, delete this import and use local one below
    _append_jsonl,
    _apply_uuid_user_ids,
    _is_main_thread,
    _list_input_files,
    _load_checkpoint,
    _load_raw_conversations,
    _save_checkpoint,
    _tqdm,
    _write_boxes_jsonl,
    logger,
    _THREAD_CTX,
)
# graph imports are deferred; loaded only when --enable-graph / --graph-from-jsonl is active

def _announce_outputs_local(stage: str, paths: List[str]):
    targets = [p for p in paths if p]
    if targets:
        logger.info("ℹ️ Stage %s will modify/append: %s", stage, ", ".join(targets))

def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
      return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "on"}:
      return True
    if v in {"0", "false", "no", "off"}:
      return False
    return default

def main(argv: List[str] | None = None, default_stage: str = "all") -> None:
    parser = argparse.ArgumentParser(description="Memory build/trace/retrieve/generate pipeline")
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override chat model name.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model name.",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=None,
        help="Override embedding base URL (default: OPENAI_EMBEDDING_BASE_URL or OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=None,
        help="Override embedding API key (default: OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--stage",
        choices=["build", "trace", "retrieve", "generate", "all"],
        default=default_stage,
        help="Which stage to run; 'all' runs the full pipeline sequentially",
    )
    parser.add_argument(
        "--graph-from-jsonl",
        action="store_true",
        help="Build graph directly from an existing memblock JSONL file (skip raw-data build).",
    )
    parser.add_argument(
        "--graph-jsonl",
        type=str,
        default=None,
        help="Path to memblock JSONL for graph build (default: final_boxes_content.jsonl).",
    )
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument(
    "--enable-graph",
    dest="enable_graph",
    action="store_true",
    help="Enable graph DB integration (Neo4j).",
    )
    graph_group.add_argument(
    "--disable-graph",
    dest="enable_graph",
    action="store_false",
    help="Disable graph DB integration (Neo4j).",
    )
    parser.set_defaults(enable_graph=_env_flag("SA_MEM_ENABLE_GRAPH", False))
    parser.add_argument(
        "--graph-max-blocks",
        type=int,
        default=-1,
        help="Max blocks to ingest from JSONL (-1 means no limit).",
    )
    parser.add_argument(
        "--graph-skip-similarity",
        action="store_true",
        help="Skip building SIMILAR edges when graph-from-jsonl is enabled.",
    )
    parser.add_argument(
        "--graph-extract-source",
        choices=["event", "raw"],
        default=Config.GRAPH_EXTRACT_SOURCE,
        help="Entity/relation extraction source for graph build: event or raw content_text.",
    )
    parser.add_argument("--build-prev-msgs", type=int, default=Config.BUILD_PREV_MSGS, help="How many previous messages to use when deciding splits")
    parser.add_argument("--answer-topn", type=str, default=str(Config.ANSWER_TOP_N), help="Number of boxes to use when answering (comma-separated)")
    parser.add_argument(
        "--text-modes",
        nargs="+",
        choices=["content", "content_trace_event", "trace_event"],
        help="Text modes for generation; default content only",
    )
    parser.add_argument("--run-id", type=str, help="Run identifier (defaults to model name)")
    parser.add_argument("--raw-data-file", type=str, default=Config.RAW_DATA_FILE, help="Raw data path (JSON list or JSONL).")
    parser.add_argument(
        "--dataset-format",
        choices=["default", "locomo", "longmemeval"],
        default="default",
        help="Input dataset format parser for build stage.",
    )
    parser.add_argument(
        "--locomo-session-prefix",
        action="store_true",
        help="When dataset-format=locomo, prefix coverage.session_id as c{conversation_index}_session{n}.",
    )
    parser.add_argument("--raw-data-dir", type=str, default=None, help="Directory containing per-uuid JSON files (each a JSON list of samples).")
    parser.add_argument("--raw-data-glob", type=str, default="*.json", help="Glob pattern inside raw-data-dir (default: *.json).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for raw-data-dir build (default: 1).")
    parser.add_argument("--uuid-start", type=int, default=None, help="Start index in raw-data-dir file list (0-based). If omitted, may resume from checkpoint.")
    parser.add_argument("--uuid-count", type=int, default=0, help="How many uuid files to process from start (0 means all).")
    parser.add_argument("--resume", action="store_true", help="Resume build from build checkpoint if present (raw-data-dir mode).")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume even if checkpoint exists.")
    parser.add_argument("--show-checkpoint", action="store_true", help="Print build checkpoint (if exists) and exit.")
    parser.add_argument("--dry-run", action="store_true", help="List raw-data-dir files and planned slice, then exit.")
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=Config.LIMIT_CONVERSATIONS if Config.LIMIT_CONVERSATIONS is not None else -1,
        help="Limit number of conversations to process (-1 means no limit).",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=Config.LIMIT_SESSIONS if Config.LIMIT_SESSIONS is not None else -1,
        help="Limit number of sessions per conversation (-1 means no limit).",
    )
    parser.add_argument(
        "--retrieval-jsonl",
        type=str,
        default=Config.SIMPLE_RETRIEVAL_JSONL,
        help="Retrieval results JSONL to use for generation",
    )
    parser.add_argument(
        "--retrieval-source",
        type=str,
        choices=["auto", "baseline", "enhanced", "full", "time_only", "unknown"],
        default="auto",
        help="Tag generation source for token_stream/stage labeling",
    )
    parser.add_argument(
        "--gen-out-jsonl",
        type=str,
        default=Config.GENERATION_RESULT_FILE,
        help="Generation output JSONL",
    )
    parser.add_argument(
        "--gen-out-csv",
        type=str,
        default=Config.GENERATION_REPORT_CSV,
        help="Generation output CSV",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Override API key (or set OPENAI_API_KEY).")
    parser.add_argument("--base-url", type=str, default=None, help="Override base URL (or set OPENAI_BASE_URL).")
    args = parser.parse_args(argv)

    if args.llm_model is not None:
        Config.LLM_MODEL = args.llm_model

    if args.embedding_model is not None:
        Config.EMBEDDING_MODEL = args.embedding_model

    Config.apply_run_id(args.run_id)
    Config.GRAPH_EXTRACT_SOURCE = str(args.graph_extract_source or "event").strip().lower()
    Config.BUILD_PREV_MSGS = max(1, args.build_prev_msgs)
    Config.TOP_K_RETRIEVE = None  # Keep full ranking

    Config.RAW_DATA_FILE = args.raw_data_file
    Config.LIMIT_CONVERSATIONS = None if args.limit_conversations == -1 else max(0, args.limit_conversations)
    Config.LIMIT_SESSIONS = None if args.limit_sessions == -1 else max(0, args.limit_sessions)
    Config.LOCOMO_SESSION_PREFIX = bool(getattr(args, "locomo_session_prefix", False))
    if args.api_key is not None:
        Config.API_KEY = args.api_key
    if args.base_url is not None:
        Config.BASE_URL = args.base_url
    if args.embedding_base_url is not None:
        Config.EMBEDDING_BASE_URL = args.embedding_base_url
    if args.embedding_api_key is not None:
        Config.EMBEDDING_API_KEY = args.embedding_api_key

    try:
        topn_list = [int(x) for x in str(args.answer_topn).split(",")]
    except ValueError:
        topn_list = [int(args.answer_topn)]
    Config.ANSWER_TOP_N = topn_list[0]

    if not (Config.API_KEY or "").strip():
        logger.warning("⚠️  OPENAI_API_KEY missing; LLM calls may fail and outputs may be low-quality.")
    worker = LLMWorker()

    text_modes = args.text_modes or Config.GEN_TEXT_MODES
    needs_trace = "content_trace_event" in text_modes or "trace_event" in text_modes

    logger.info("ℹ️ Using run_id=%s, output_dir=%s", Config.RUN_ID, Config.OUTPUT_DIR)

    if args.show_checkpoint:
        ck = _load_checkpoint(Config.BUILD_CHECKPOINT_FILE)
        if ck is None:
            print("No checkpoint found:", Config.BUILD_CHECKPOINT_FILE)
        else:
            print(json.dumps(ck, ensure_ascii=False, indent=2))
        return

    if args.enable_graph or args.graph_from_jsonl:
        from graph_storage import GraphConfig  # deferred: only needed for graph mode
        graph_cfg = GraphConfig(enable_graph=bool(args.enable_graph))
    else:
        graph_cfg = None
        logger.info("ℹ️ Graph DB disabled by config (--disable-graph or SA_MEM_ENABLE_GRAPH=0).")
    # Prefer using local announce if extractor no longer provides it
    announce = _announce_outputs if "_announce_outputs" in globals() else _announce_outputs_local

    # ---------------- BUILD ----------------
    if args.stage in ("build", "all"):
        announce("build", [Config.FINAL_CONTENT_FILE, Config.BUILD_TRACE_FILE, Config.TOKEN_LOG_FILE, Config.BUILD_STATS_FILE, Config.VECTOR_DIR])
        TokenAnalyzer.stage_stats["build"] = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
        if args.graph_from_jsonl:
            from build_impl_graph import build_graph_from_jsonl  # deferred: graph-only path
            jsonl_path = args.graph_jsonl or Config.FINAL_CONTENT_FILE
            max_blocks = None if args.graph_max_blocks == -1 else max(0, args.graph_max_blocks)
            build_graph_from_jsonl(
                jsonl_path=jsonl_path,
                worker=worker,
                graph_config=graph_cfg,
                max_blocks=max_blocks,
                skip_similarity=bool(args.graph_skip_similarity),
            )
            if args.stage == "build":
                return
        builder = MemoryBuilder(worker)
        use_locomo = args.dataset_format == "locomo"
        use_longmemeval = args.dataset_format == "longmemeval"

        if args.raw_data_dir:
            files = _list_input_files(args.raw_data_dir, args.raw_data_glob)
            ck = _load_checkpoint(Config.BUILD_CHECKPOINT_FILE) if (args.resume and not args.no_resume) else None
            default_start = int(ck.get("next_file_index", 0)) if ck else 0
            start_idx = args.uuid_start if args.uuid_start is not None else default_start
            start_idx = max(0, min(start_idx, len(files)))
            count = args.uuid_count
            end_idx = len(files) if not count or count <= 0 else min(len(files), start_idx + count)
            planned = files[start_idx:end_idx]

            if args.dry_run:
                print(f"Found {len(files)} file(s) in {args.raw_data_dir} ({args.raw_data_glob})")
                print(f"Planned slice: start={start_idx} count={count or 'ALL'} end={end_idx} (total={len(planned)})")
                for i, fp in enumerate(planned, start=start_idx):
                    print(f"[{i}] {fp}")
                return

            ck_obj: Dict[str, Any] = ck or {
                "run_id": Config.RUN_ID,
                "raw_data_dir": args.raw_data_dir,
                "raw_data_glob": args.raw_data_glob,
                "created_at": datetime.now().isoformat(),
                "next_file_index": start_idx,
                "processed_files": [],
                "user_id_next": 0,
            }

            processed_prev = set(ck_obj.get("processed_files") or []) if isinstance(ck_obj.get("processed_files"), list) else set()
            tmp_dir = os.path.join(Config.OUTPUT_DIR, "tmp_build")
            os.makedirs(tmp_dir, exist_ok=True)

            def _count_sessions_in_raw_list(raw_list: List[Dict[str, Any]]) -> int:
                if Config.LIMIT_CONVERSATIONS is not None:
                    raw_list = raw_list[: Config.LIMIT_CONVERSATIONS]
                total = 0
                for item in raw_list:
                    if use_longmemeval:
                        sessions = (item or {}).get("haystack_sessions", [])
                        if not isinstance(sessions, list):
                            continue
                        count = len(sessions)
                        if Config.LIMIT_SESSIONS is not None:
                            count = min(count, Config.LIMIT_SESSIONS)
                        total += count
                    else:
                        conv = (item or {}).get("conversation", {})
                        if not isinstance(conv, dict):
                            continue
                        if use_locomo:
                            keys = sorted(
                                [k for k in conv.keys() if isinstance(k, str) and re.fullmatch(r"session_\d+", k)],
                                key=lambda x: int(x.split("_")[1]),
                            )
                        else:
                            keys = sorted(
                                [k for k in conv.keys() if k.startswith("session_") and len(k) < 12],
                                key=lambda x: int(x.split("_")[1]),
                            )
                        session_keys = keys[: Config.LIMIT_SESSIONS] if Config.LIMIT_SESSIONS is not None else keys
                        total += len(session_keys)
                return total

            def _count_sessions_in_file(file_path: str) -> int:
                try:
                    raw_list = _load_raw_conversations(file_path)
                    return _count_sessions_in_raw_list(raw_list)
                except Exception:
                    return 0

            def _job(file_index: int, file_path: str) -> Tuple[int, str, str, int, Dict[str, Any]]:
                raw_list = _load_raw_conversations(file_path)
                source_id = pathlib.Path(file_path).stem
                uuid8 = (source_id or "")[:8]
                for item in raw_list:
                    if isinstance(item, dict) and "_source_id" not in item:
                        item["_source_id"] = source_id

                local_worker = LLMWorker()
                local_builder = MemoryBuilder(local_worker)

                if uuid8:
                    _THREAD_CTX.user_id_map = {i: (uuid8 if i == 0 else f"{uuid8}_{i}") for i in range(len(raw_list))}
                else:
                    _THREAD_CTX.user_id_map = None

                def _on_session_done():
                    session_q.put(1)

                try:
                    if use_longmemeval and hasattr(local_builder, "build_all_longmemeval"):
                        boxes = local_builder.build_all_longmemeval(
                            raw_list_override=raw_list,
                            user_id_start=0,
                            write_incremental=False,
                            on_session_done=_on_session_done,
                        )
                    elif use_locomo and hasattr(local_builder, "build_all_locomo"):
                        boxes = local_builder.build_all_locomo(
                            raw_list_override=raw_list,
                            user_id_start=0,
                            write_incremental=False,
                            on_session_done=_on_session_done,
                        )
                    else:
                        boxes = local_builder.build_all(
                            raw_list_override=raw_list,
                            user_id_start=0,
                            write_incremental=False,
                            on_session_done=_on_session_done,
                        )
                    if uuid8:
                        _apply_uuid_user_ids(boxes, uuid8=uuid8, user_id_start=0, user_count=len(raw_list))
                    out_tmp = os.path.join(tmp_dir, f"{source_id}.jsonl")
                    _write_boxes_jsonl(out_tmp, boxes)
                    local_stats = {"boxes": local_builder.total_boxes, "messages": sum(local_builder.msg_counts) if local_builder.msg_counts else 0}
                    return file_index, file_path, out_tmp, len(raw_list), local_stats
                finally:
                    try:
                        _THREAD_CTX.user_id_map = None
                    except Exception:
                        pass

            to_process: List[Tuple[int, str]] = []
            to_skip: List[Tuple[int, str]] = []
            for file_index, file_path in enumerate(planned, start=start_idx):
                source_id = pathlib.Path(file_path).stem
                out_tmp = os.path.join(tmp_dir, f"{source_id}.jsonl")
                if (args.resume and not args.no_resume) and (file_path in processed_prev) and os.path.exists(out_tmp):
                    to_skip.append((file_index, file_path))
                else:
                    to_process.append((file_index, file_path))

            if args.workers and args.workers > 1:
                logger.info("⚡ Parallel build enabled: workers=%s (files=%s, skip=%s)", args.workers, len(to_process), len(to_skip))
                results: List[Tuple[int, str, str, int, Dict[str, Any]]] = []

                session_q: "queue.Queue[int]" = queue.Queue()
                total_sessions = sum(_count_sessions_in_file(fp) for _, fp in to_process)

                try:
                    from tqdm import tqdm  # type: ignore
                except Exception:
                    tqdm = None

                with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
                    futs = [ex.submit(_job, fi, fp) for fi, fp in to_process]
                    pending = set(futs)
                    done_boxes = 0
                    done_files = 0

                    p_files = tqdm(total=len(futs), desc="BUILD uuid files") if tqdm else None
                    p_sessions = tqdm(total=total_sessions, desc="BUILD sessions", leave=False) if (tqdm and total_sessions) else None

                    try:
                        while pending:
                            drained = 0
                            while True:
                                try:
                                    session_q.get_nowait()
                                    drained += 1
                                except queue.Empty:
                                    break
                            if drained and p_sessions:
                                p_sessions.update(drained)

                            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                            for fut in done:
                                r = fut.result()
                                results.append(r)
                                done_files += 1
                                stats = r[4] or {}
                                try:
                                    done_boxes += int(stats.get("boxes", 0) or 0)
                                except Exception:
                                    pass
                                if p_files:
                                    p_files.update(1)
                                    p_files.set_postfix({"files": f"{done_files}/{len(futs)}", "boxes": done_boxes})
                    finally:
                        drained = 0
                        while True:
                            try:
                                session_q.get_nowait()
                                drained += 1
                            except queue.Empty:
                                break
                        if drained and p_sessions:
                            p_sessions.update(drained)
                        if p_sessions:
                            p_sessions.close()
                        if p_files:
                            p_files.close()

                results_by_index = {r[0]: r for r in results}
                appended_boxes = 0
                for file_index, file_path in to_process:
                    _, _, out_tmp, _, _ = results_by_index[file_index]
                    appended_boxes += _append_jsonl(Config.FINAL_CONTENT_FILE, out_tmp)

                ck_obj["updated_at"] = datetime.now().isoformat()
                ck_obj["next_file_index"] = end_idx
                ck_obj["last_file"] = planned[-1] if planned else None
                processed = ck_obj.get("processed_files")
                if not isinstance(processed, list):
                    processed = []
                    ck_obj["processed_files"] = processed
                for _, fp in to_process:
                    processed.append(fp)
                ck_obj["appended_boxes_last_run"] = appended_boxes
                _save_checkpoint(Config.BUILD_CHECKPOINT_FILE, ck_obj)
            else:
                user_id_next = int(ck_obj.get("user_id_next", 0) or 0)
                for file_index, file_path in _tqdm(list(enumerate(planned, start=start_idx)), total=len(planned), desc="BUILD"):
                    try:
                        raw_list = _load_raw_conversations(file_path)
                        source_id = pathlib.Path(file_path).stem
                        uuid8 = (source_id or "")[:8]
                        for item in raw_list:
                            if isinstance(item, dict) and "_source_id" not in item:
                                item["_source_id"] = source_id

                        if uuid8:
                            _THREAD_CTX.user_id_map = {i: (uuid8 if i == 0 else f"{uuid8}_{i}") for i in range(len(raw_list))}
                        else:
                            _THREAD_CTX.user_id_map = None
                        try:
                            if use_longmemeval and hasattr(builder, "build_all_longmemeval"):
                                boxes = builder.build_all_longmemeval(raw_list_override=raw_list, user_id_start=0, write_incremental=False)
                            elif use_locomo and hasattr(builder, "build_all_locomo"):
                                boxes = builder.build_all_locomo(raw_list_override=raw_list, user_id_start=0, write_incremental=False)
                            else:
                                boxes = builder.build_all(raw_list_override=raw_list, user_id_start=0, write_incremental=False)
                            if uuid8:
                                _apply_uuid_user_ids(boxes, uuid8=uuid8, user_id_start=0, user_count=len(raw_list))
                        finally:
                            try:
                                _THREAD_CTX.user_id_map = None
                            except Exception:
                                pass

                        builder.save_incremental(boxes, append=True)
                        user_id_next += len(raw_list)

                        ck_obj["updated_at"] = datetime.now().isoformat()
                        ck_obj["next_file_index"] = file_index + 1
                        ck_obj["last_file"] = file_path
                        ck_obj["user_id_next"] = user_id_next
                        processed = ck_obj.get("processed_files")
                        if isinstance(processed, list):
                            processed.append(file_path)
                        _save_checkpoint(Config.BUILD_CHECKPOINT_FILE, ck_obj)
                    except Exception as e:
                        ck_obj["updated_at"] = datetime.now().isoformat()
                        ck_obj["error"] = str(e)
                        ck_obj["error_file"] = file_path
                        _save_checkpoint(Config.BUILD_CHECKPOINT_FILE, ck_obj)
                        raise
        else:
            if use_longmemeval and hasattr(builder, "build_all_longmemeval"):
                boxes = builder.build_all_longmemeval()
            elif use_locomo and hasattr(builder, "build_all_locomo"):
                boxes = builder.build_all_locomo()
            else:
                boxes = builder.build_all()
            if not Config.CHECKPOINT_EVERY_SAMPLE:
                builder.save(boxes)

        builder.summarize_and_log()
        logger.info("✅ Build done")

    # ---------------- TRACE ----------------
    if args.stage in ("trace", "all"):
        if needs_trace:
            announce("trace", [Config.TIME_TRACE_FILE, Config.TRACE_PROMPT_LOG_FILE, Config.VECTOR_DIR])
            linker = TraceLinker(worker, trace_metrics=Config.TRACE_METRICS)
            linker.run()
        else:
            logger.info("ℹ️ Trace skipped because text_mode excludes trace events.")

    # ---------------- RETRIEVE ----------------
    if args.stage in ("retrieve", "all"):
        announce("retrieve", [Config.SIMPLE_RETRIEVAL_JSONL, Config.SIMPLE_RETRIEVAL_CSV, Config.VECTOR_DIR])
        retr = SimpleRetriever(worker, top_k=Config.TOP_K_RETRIEVE)
        retr.run(Config.SIMPLE_RETRIEVAL_JSONL, Config.SIMPLE_RETRIEVAL_CSV)

    # ---------------- GENERATE ----------------
    if args.stage in ("generate", "all"):
        announce("generate", [args.gen_out_jsonl, args.gen_out_csv, Config.GEN_SUMMARY_FILE, Config.TOKEN_LOG_FILE])
        if needs_trace and not os.path.exists(Config.TIME_TRACE_FILE):
            logger.warning("Trace file missing but trace text_mode requested; run trace stage first.")

        generator = AnswerGenerator(
            worker,
            answer_topn=topn_list,
            text_modes=text_modes,
            stage_label="gen",
        )
        generator.run(
            args.retrieval_jsonl,
            args.gen_out_jsonl,
            args.gen_out_csv,
            retrieval_source=args.retrieval_source,
        )


if __name__ == "__main__":
    main()