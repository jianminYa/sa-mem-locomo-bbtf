import json
import os
import re
from typing import Any, Dict, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity

from graph_entities_extractor import GraphEntitiesExtractor, OllamaToolCaller, OpenAIToolCaller
from graph_storage import (
    GraphConfig,
    MemgraphEventGraph,
    build_graph_payload_from_memblock,
    extract_and_store_event_entities,
    extract_and_store_block_entities,
)


def _mx():
    """
    Lazy import to avoid circular import:
    memblock_extractor imports build_impl, but build_impl must not import memblock_extractor at import-time.
    """
    import memblock_extractor as mx  # local import by design

    return mx


def _load_event_vector_cache(store: Any) -> Dict[str, List[float]]:
    cache: Dict[str, List[float]] = {}
    for key, fields in (store.data or {}).items():
        if not isinstance(key, str) or not key.startswith("graph_event_"):
            continue
        if not isinstance(fields, dict):
            continue
        vec = fields.get("event")
        if isinstance(vec, list) and vec:
            event_id = key[len("graph_event_") :]
            if event_id:
                cache[event_id] = vec
    return cache


def _link_similar_events_global(
    *,
    graph: MemgraphEventGraph,
    user_id: str,
    event_id: str,
    event_vec: List[float],
    existing_event_vecs: Dict[str, List[float]],
    threshold: float,
    limit: int,
) -> None:
    if not existing_event_vecs or not event_vec:
        return
    pairs: List[Tuple[str, float]] = []
    for other_id, other_vec in existing_event_vecs.items():
        if other_id == event_id:
            continue
        try:
            score = float(cosine_similarity([event_vec], [other_vec])[0][0])
        except Exception:
            score = -1.0
        if score >= threshold:
            pairs.append((other_id, score))
    if not pairs:
        return
    pairs.sort(key=lambda x: x[1], reverse=True)
    for other_id, score in pairs[: max(0, limit)]:
        graph.link_similar_events(
            user_id=str(user_id),
            a_id=event_id,
            b_id=other_id,
            score=score,
        )


def build_graph_from_jsonl(
    *,
    jsonl_path: str,
    worker: Any,
    graph_config: GraphConfig | None = None,
    max_blocks: int | None = None,
    skip_similarity: bool = False,
) -> None:
    mx = _mx()
    cfg = graph_config or GraphConfig()
    if not cfg.enable_graph:
        mx.logger.info("ℹ️ Graph disabled via config; skip graph build")
        return
    if not jsonl_path or not os.path.exists(jsonl_path):
        mx.logger.error("❌ jsonl not found: %s", jsonl_path)
        return

    graph = MemgraphEventGraph(
        url=cfg.memgraph_url,
        username=cfg.memgraph_username,
        password=cfg.memgraph_password,
    )
    llm_provider = str(os.getenv("GRAPH_LLM_PROVIDER") or "openai").strip().lower()
    graph_prompt_extra = os.getenv("GRAPH_RELATION_PROMPT_EXTRA")
    if llm_provider == "ollama":
        ollama_model = os.getenv("GRAPH_LLM_MODEL") or mx.Config.OLLAMA_LLM_MODEL
        ollama_base_url = os.getenv("GRAPH_LLM_BASE_URL") or mx.Config.OLLAMA_BASE_URL
        tool_caller = OllamaToolCaller(model=ollama_model, base_url=ollama_base_url)
    else:
        openai_model = os.getenv("GRAPH_LLM_MODEL") or "Qwen/Qwen3-30B-A3B-Instruct-2507"
        openai_base_url = os.getenv("GRAPH_LLM_BASE_URL") or "http://localhost:8000/v1"
        openai_api_key = os.getenv("GRAPH_LLM_API_KEY") or mx.Config.API_KEY or os.getenv("OPENAI_API_KEY")
        tool_caller = OpenAIToolCaller(
            model=openai_model,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )
    raw_run_id = str(mx.Config.RUN_ID or "run").strip()
    safe_run_id = re.sub(r"[^A-Za-z0-9_\-]+", "_", raw_run_id).strip("_") or "run"
    entity_list_dir = mx.Config.OUTPUT_DIR
    entity_list_filename = f"{safe_run_id}_entitylist.json"
    extractor = GraphEntitiesExtractor(
        llm=tool_caller,
        custom_prompt=graph_prompt_extra,
        use_heuristic_relations=False,
        drop_weak_relations=True,
        entity_list_dir=entity_list_dir,
        entity_list_filename=entity_list_filename,
    )

    extract_source = str(mx.Config.GRAPH_EXTRACT_SOURCE or "event").strip().lower()

    store_by_user: Dict[str, Any] = {}
    vec_cache_by_user: Dict[str, Dict[str, List[float]]] = {}

    mx.logger.info("ℹ️ Graph build from jsonl: %s", jsonl_path)
    processed = 0
    log_path = os.path.join(os.path.dirname(jsonl_path), "graph_input.log")
    with open(jsonl_path, "r", encoding="utf-8") as f, open(log_path, "a", encoding="utf-8") as log_f:
        for line in f:
            if max_blocks is not None and processed >= max_blocks:
                break
            raw = (line or "").strip()
            if not raw:
                continue
            try:
                block = json.loads(raw)
            except Exception:
                continue
            user_id = str(block.get("user_id") or "")
            if not user_id:
                continue
            block_id = block.get("block_id")
            events = block.get("events") or []
            features = block.get("features", {}) or {}
            content_text = str(features.get("content_text") or "").strip()
            block_key = f"{user_id}|{block_id}" if block_id is not None else f"{user_id}|unknown"

            store = store_by_user.get(user_id)
            if store is None:
                store = mx.EmbeddingStore(worker, user_id)
                store_by_user[user_id] = store
                vec_cache_by_user[user_id] = _load_event_vector_cache(store)
            event_vec_cache = vec_cache_by_user.get(user_id, {})

            if extract_source == "raw" and content_text:
                stats = extract_and_store_block_entities(
                    event_graph=graph,
                    extractor=extractor,
                    block_key=block_key,
                    block_id=block_id,
                    user_id=str(user_id),
                    content_text=content_text,
                )
                log_f.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "block_id": block_id,
                            "event_id": None,
                            "entity_tool_calls": stats.get("entity_tool_calls", 0),
                            "relation_tool_calls": stats.get("relation_tool_calls", 0),
                            "entity_count": stats.get("entity_count", 0),
                            "relation_count": stats.get("relation_count", 0),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            for ev in events:
                payload = build_graph_payload_from_memblock(block=block, event=ev, user_id=user_id)
                payload["created_at"] = payload.get("created_at") or mx.datetime.utcnow().isoformat() + "Z"
                memory_id = str(ev.get("event_id") or "")
                if memory_id:
                    graph.upsert_event_node(memory_id=memory_id, payload=payload)
                    if extract_source != "raw":
                        stats = extract_and_store_event_entities(
                            event_graph=graph,
                            extractor=extractor,
                            event_id=memory_id,
                            user_id=str(user_id),
                            event_description=str(ev.get("description") or ""),
                        )
                        log_f.write(
                            json.dumps(
                                {
                                    "user_id": user_id,
                                    "block_id": block_id,
                                    "event_id": memory_id,
                                    "entity_tool_calls": stats.get("entity_tool_calls", 0),
                                    "relation_tool_calls": stats.get("relation_tool_calls", 0),
                                    "entity_count": stats.get("entity_count", 0),
                                    "relation_count": stats.get("relation_count", 0),
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                desc = str(ev.get("description") or "").strip()
                if desc and memory_id:
                    vec = store.get_vector(
                        f"graph_event_{memory_id}",
                        "event",
                        desc,
                        note=f"U{user_id}_E{memory_id}_graph",
                    )
                    if vec:
                        if not skip_similarity:
                            threshold = float(getattr(cfg, "graph_similarity_threshold", 0.8) or 0.8)
                            limit = int(getattr(cfg, "graph_similar_limit", 20) or 20)
                            _link_similar_events_global(
                                graph=graph,
                                user_id=str(user_id),
                                event_id=memory_id,
                                event_vec=vec,
                                existing_event_vecs=event_vec_cache,
                                threshold=threshold,
                                limit=limit,
                            )
                        event_vec_cache[memory_id] = vec

                store.flush()
                processed += 1
    graph.close()
    mx.logger.info("✅ Graph build done. Blocks processed: %d", processed)


class TopicClusterManager:
    def __init__(self, worker: Any):
        self.worker = worker

    def process_new_box(self, new_box: Dict[str, Any], user_id: Any):
        mx = _mx()
        prefix = f"U{user_id}_B{new_box['box_id']}"
        content_str = self._get_content_str(new_box)

        # ✅ get session window FIRST (so we can fill prompt placeholders)
        bg = new_box.get("background_info", {}) or {}
        session_start_time = bg.get("start_time")
        session_end_time = bg.get("end_time", session_start_time)

        # ✅ PASS 1: Extract mentions (strings only, no classification)
        ps1_prompt = (
            mx.Config.PROMPT_DIALOG_EXTRACT
            .replace("{text}", content_str)
        )

        ps1_raw = self.worker.chat_completion(
            ps1_prompt,
            note=f"{prefix}_NotePS1_Extract",
            json_mode=True,
            enable_functions=True,  # Enable temporal resolution function calling
            extra={"prompt_tokens_est": self.worker.count_tokens(ps1_prompt), "stage": "build"},
        )

        # 🔍 [DEBUG FIX] Detect empty response and log warning
        if not ps1_raw or not str(ps1_raw).strip():
             mx.logger.warning(f"⚠️ [WARN] Box {new_box['box_id']}: LLM returned EMPTY response (Pass 1). Likely timeout/rate-limit. Fallback triggered.")
        elif new_box["box_id"] < 5:
             # Verify LLM output for first few boxes
             mx.logger.info(f"🔍 [DEBUG] Box {new_box['box_id']} Pass 1 raw output (first 500 chars): {str(ps1_raw)[:500]}...")

        topic = ""
        keywords_txt = ""
        mentions: List[str] = []  # Pass 1 returns string mentions
        try:
            d = mx.json.loads(ps1_raw)
            topic = str(d.get("topic", "") or "").strip()
            kws = d.get("keywords", [])
            if isinstance(kws, list):
                keywords_txt = ", ".join([str(k).strip() for k in kws if str(k).strip()])
            else:
                keywords_txt = str(kws).strip()

            # Parse both prompt variants: Halumem uses "mentions", LoCoMo uses
            # "explicit_mentions".
            mns = d.get("mentions", d.get("explicit_mentions", []))
            if isinstance(mns, list):
                mentions = [str(m).strip() for m in mns if str(m).strip()]
            else:
                mentions = []
        except Exception as e:
            # 🔍 [DEBUG FIX] Catch JSON parse failures
            if ps1_raw and str(ps1_raw).strip():
                mx.logger.error(f"❌ [ERROR] Box {new_box['box_id']} Pass 1 JSON parse failed: {e}. Raw: {str(ps1_raw)[:50]}")
            pass

        # ✅ PASS 2: Classify mentions and assign temporal metadata
        events: List[Any] = []  # Will hold dict events (with type, start_time, end_time)
        if mentions:
            if mx.Config.ENABLE_EVENT_CLASSIFICATION:
                # Use LLM for event classification
                mentions_json = mx.json.dumps(mentions, ensure_ascii=False)
                ps2_prompt = (
                    mx.Config.PROMPT_DIALOG_CLASSIFICATION
                    .replace("{session_start_time}", str(session_start_time or "Unknown"))
                    .replace("{session_end_time}", str(session_end_time or session_start_time or "Unknown"))
                    .replace("{mentions_json}", mentions_json)
                )

                ps2_raw = self.worker.chat_completion(
                    ps2_prompt,
                    note=f"{prefix}_NotePS2_Classify",
                    json_mode=True,
                    extra={"prompt_tokens_est": self.worker.count_tokens(ps2_prompt), "stage": "build"},
                )

                if not ps2_raw or not str(ps2_raw).strip():
                    mx.logger.warning(f"⚠️ [WARN] Box {new_box['box_id']}: LLM returned EMPTY response (Pass 2). Fallback triggered.")
                elif new_box["box_id"] < 5:
                    mx.logger.info(f"🔍 [DEBUG] Box {new_box['box_id']} Pass 2 raw output (first 500 chars): {str(ps2_raw)[:500]}...")

                try:
                    d2 = mx.json.loads(ps2_raw)
                    evs = d2.get("explicit_mentions", [])
                    if isinstance(evs, list):
                        events = [ev for ev in evs if ev]
                    else:
                        events = []
                except Exception as e:
                    if ps2_raw and str(ps2_raw).strip():
                        mx.logger.error(f"❌ [ERROR] Box {new_box['box_id']} Pass 2 JSON parse failed: {e}. Raw: {str(ps2_raw)[:50]}")
                    pass
            else:
                # Use heuristic fallback (no LLM cost)
                mx.logger.info(f"🔍 Event classification DISABLED, using heuristic fallback for box {new_box['box_id']}")
                events = []
                for mention in mentions:
                    # Create simple event with heuristic type
                    event_type = mx.infer_event_type(mention) if hasattr(mx, 'infer_event_type') else "MISC"
                    # Use session window as fallback for temporal metadata
                    events.append({
                        "description": mention,
                        "type": event_type,
                        "start_time": str(session_start_time or "Unknown"),
                        "end_time": str(session_end_time or session_start_time or "Unknown")
                    })

        topic_kw_text = f"{topic} {keywords_txt}".strip()

        fallback_event_text = topic_kw_text or topic or keywords_txt
        if not fallback_event_text:
            for m in new_box.get("content", []) or []:
                txt = str(m.get("text", "") or "").strip()
                if txt:
                    fallback_event_text = txt[:120]
                    break
        fallback_event_text = fallback_event_text or "conversation"

        # ✅ IMPORTANT: normalize as OBJECTS to preserve LLM-provided `type`
        events_obj = mx._normalize_event_objs(
            events,
            session_start_time,
            session_end_time,
            fallback_text=fallback_event_text,
        )
        events_obj = mx._dedupe_and_filter_event_objs(
            events_obj,
            drop_texts=[topic_kw_text, fallback_event_text],
        )

        new_box["runtime_info"] = {
            "topic": topic,
            "topic_kw_text": topic_kw_text,
            "keywords": keywords_txt,
            "events": events_obj,
        }

        mx.TraceLogger.log(
            mx.Config.BUILD_TRACE_FILE,
            {
                "type": "box_created",
                "user_id": user_id,
                "box_id": new_box["box_id"],
                "content_preview": content_str[:100],
                "extracted": {
                    "topic": topic,
                    "keywords": keywords_txt,
                    "events": [mx._event_any_to_line(x) for x in (events_obj or [])],
                },
            },
        )

    def _get_content_str(self, box: Dict[str, Any]) -> str:
        bg = box.get("background_info", {})
        header = "Session window: " + str(bg.get("start_time", "Unknown")) + " -> " + str(
            bg.get("end_time", bg.get("start_time", "Unknown"))
        )

        # ✅ Halumem: include persona_info so LLM can use the user's name in extracted events
        persona_info = str(box.get("persona_info", "") or "").strip()
        persona_line = f"Persona info: {persona_info}" if persona_info else ""

        lines: List[str] = []
        for m in box.get("content", []) or []:
            ts = m.get("time", "")
            lines.append(f"{ts} {m.get('role')}: {m.get('text')}")

        parts: List[str] = [header]
        if persona_line:
            parts.append(persona_line)
        parts.extend(lines)
        return "\n".join(parts)



class MemoryBuilder:
    def __init__(self, worker: Any, graph_config: GraphConfig | None = None):
        self.worker = worker
        self.graph_config = graph_config or GraphConfig()
        self.event_graph: MemgraphEventGraph | None = None
        self.entity_extractor: GraphEntitiesExtractor | None = None
        if self.graph_config.enable_graph:
            self.event_graph = MemgraphEventGraph(
                url=self.graph_config.memgraph_url,
                username=self.graph_config.memgraph_username,
                password=self.graph_config.memgraph_password,
            )
            graph_prompt_extra = os.getenv("GRAPH_RELATION_PROMPT_EXTRA")
            graph_provider = str(os.getenv("GRAPH_LLM_PROVIDER") or "openai").strip().lower()
            if graph_provider == "ollama":
                ollama_model = os.getenv("GRAPH_LLM_MODEL") or mx.Config.OLLAMA_LLM_MODEL
                ollama_base_url = os.getenv("GRAPH_LLM_BASE_URL") or mx.Config.OLLAMA_BASE_URL
                llm = OllamaToolCaller(model=ollama_model, base_url=ollama_base_url)
            else:
                openai_model = os.getenv("GRAPH_LLM_MODEL") or "Qwen/Qwen3-30B-A3B-Instruct-2507"
                openai_base_url = os.getenv("GRAPH_LLM_BASE_URL") or "http://localhost:8000/v1"
                openai_api_key = os.getenv("GRAPH_LLM_API_KEY") or mx.Config.API_KEY or os.getenv("OPENAI_API_KEY")
                llm = OpenAIToolCaller(
                    model=openai_model,
                    api_key=openai_api_key,
                    base_url=openai_base_url,
                )
            self.entity_extractor = GraphEntitiesExtractor(
                llm=llm,
                custom_prompt=graph_prompt_extra,
                use_heuristic_relations=False,
                drop_weak_relations=True,
            )
        self.cluster: TopicClusterManager | None = None
        self.boxes: List[Dict[str, Any]] = []
        self.msgs: List[Dict[str, Any]] = []
        self.bid = 0
        self.token_ratios: List[float] = []
        self.msg_counts: List[int] = []
        self.box_token_pairs: List[Tuple[int, int]] = []
        self.total_boxes: int = 0

    def build_all(
        self,
        raw_list_override: List[Dict[str, Any]] | None = None,
        user_id_start: int = 0,
        write_incremental: bool | None = None,
        on_session_done=None,
    ):
        mx = _mx()
        all_samples_boxes: List[Dict[str, Any]] = []
        raw_list = raw_list_override if raw_list_override is not None else mx._load_raw_conversations(mx.Config.RAW_DATA_FILE)

        if mx.Config.LIMIT_CONVERSATIONS is not None:
            raw_list = raw_list[: mx.Config.LIMIT_CONVERSATIONS]

        if mx._is_main_thread():
            mx.logger.info("🏗️  [BUILD] Processing %s Conversations...", len(raw_list))

        do_write_incremental = mx.Config.CHECKPOINT_EVERY_SAMPLE if write_incremental is None else bool(write_incremental)

        for i, conversation_data in enumerate(mx._tqdm(raw_list, total=len(raw_list), desc="BUILD conversations")):
            raw_user_id = (conversation_data or {}).get("user_id")
            user_id = raw_user_id if raw_user_id is not None else (user_id_start + i)

            if mx._is_main_thread():
                display_id = user_id
                try:
                    user_id_map = getattr(mx._THREAD_CTX, "user_id_map", None)
                    if user_id_map and user_id in user_id_map:
                        display_id = user_id_map[user_id]
                except Exception:
                    display_id = user_id
                mx.logger.info("   Building Sample %s...", display_id)

            self.cluster = TopicClusterManager(self.worker)
            self.boxes = []
            self.msgs = []
            # ✅ Don't reset bid - keep it globally unique across all conversations
            # self.bid = 0

            conv = (conversation_data or {}).get("conversation", {}) or {}

            # ✅ Halumem: username usually in persona_info
            persona_info = str(conv.get("persona_info", "") or "").strip()

            def _extract_name_from_persona_info(s: str) -> str | None:
                if not s:
                    return None
                m = mx.re.search(r"\bName\s*:\s*([^;\n\r]+)", s, flags=mx.re.IGNORECASE)
                if not m:
                    return None
                name = m.group(1).strip()
                return name or None

            persona_name = _extract_name_from_persona_info(persona_info)

            static_meta = {
                "speaker_a": conv.get("speaker_a", "A"),
                "speaker_b": conv.get("speaker_b", "B"),
                "persona_info": persona_info,   # ✅ pass through
                "persona_name": persona_name,   # ✅ pass through
            }

            keys = sorted(
                [k for k in conv.keys() if k.startswith("session_") and len(k) < 12],
                key=lambda x: int(x.split("_")[1]),
            )
            session_keys = keys[: mx.Config.LIMIT_SESSIONS] if mx.Config.LIMIT_SESSIONS is not None else keys

            current_session_id = None
            for k in mx._tqdm(session_keys, total=len(session_keys), desc=f"U{user_id} sessions", leave=False):
                t_key = f"{k}_date_time"  # legacy
                start_key = f"{k}_start_time"
                end_key = f"{k}_end_time"
                session_start_time = conv.get(start_key, conv.get(t_key, "Unknown"))
                session_end_time = conv.get(end_key, session_start_time)

                if "json" in str(session_start_time):
                    session_start_time = "Unknown"
                if "json" in str(session_end_time):
                    session_end_time = session_start_time

                session_msgs = conv.get(k)
                if not isinstance(session_msgs, list):
                    if on_session_done is not None:
                        try:
                            on_session_done()
                        except Exception:
                            pass
                    continue

                current_session_id = k
                for idx, m in enumerate(session_msgs, start=1):
                    if not isinstance(m, dict):
                        continue
                    msg_time = m.get("time") or m.get("timestamp") or session_start_time
                    msg = {
                        "role": m.get("speaker"),
                        "text": str(m.get("text", "") or "").strip(),
                        "time": msg_time,
                        "_temp_session_start_time": session_start_time,
                        "_temp_session_end_time": session_end_time,
                    }
                    self._process(msg, static_meta, user_id, current_session_id, idx)

                if on_session_done is not None:
                    try:
                        on_session_done()
                    except Exception:
                        pass

            self._seal(static_meta, user_id, current_session_id)
            all_samples_boxes.extend(self.boxes)

            if do_write_incremental:
                self.save_incremental(self.boxes, append=True)

        return all_samples_boxes

    def build_all_locomo(
        self,
        raw_list_override: List[Dict[str, Any]] | None = None,
        user_id_start: int = 0,
        write_incremental: bool | None = None,
        on_session_done=None,
    ):
        """Build blocks for LoCoMo-like samples.

        Expected sample format (top-level list):
        {
            "qa": [...],
            "conversation": {...},
            "event_summary": {...},
            "observation": {...},
            "session_summary": {...},
            "sample_id": "conv-26"
        }

        Notes:
        - Only `conversation` is required for block construction.
        - `session_i_start_time` / `session_i_end_time` are optional.
          If absent, falls back to `session_i_date_time`.
        - Message `time`/`timestamp` is optional and falls back to session time.
        """
        mx = _mx()
        all_samples_boxes: List[Dict[str, Any]] = []
        raw_list = raw_list_override if raw_list_override is not None else mx._load_raw_conversations(mx.Config.RAW_DATA_FILE)

        if mx.Config.LIMIT_CONVERSATIONS is not None:
            raw_list = raw_list[: mx.Config.LIMIT_CONVERSATIONS]

        if mx._is_main_thread():
            mx.logger.info("🏗️  [BUILD-LOCOMO] Processing %s Conversations...", len(raw_list))

        do_write_incremental = mx.Config.CHECKPOINT_EVERY_SAMPLE if write_incremental is None else bool(write_incremental)

        for i, sample in enumerate(mx._tqdm(raw_list, total=len(raw_list), desc="BUILD locomo conversations")):
            user_id = f"{i}"
            sample_id = str((sample or {}).get("sample_id", "") or "").strip()
            conv_idx_1based = i + 1
            use_locomo_session_prefix = bool(getattr(mx.Config, "LOCOMO_SESSION_PREFIX", False))

            if mx._is_main_thread():
                display_id = sample_id or user_id
                try:
                    user_id_map = getattr(mx._THREAD_CTX, "user_id_map", None)
                    if user_id_map and user_id in user_id_map:
                        display_id = user_id_map[user_id]
                except Exception:
                    pass
                mx.logger.info("   Building LoCoMo Sample %s...", display_id)

            self.cluster = TopicClusterManager(self.worker)
            self.boxes = []
            self.msgs = []

            conv = (sample or {}).get("conversation", {}) or {}
            if not isinstance(conv, dict):
                if on_session_done is not None:
                    try:
                        on_session_done()
                    except Exception:
                        pass
                continue

            persona_info = str(conv.get("persona_info", "") or "").strip()

            def _extract_name_from_persona_info(s: str) -> str | None:
                if not s:
                    return None
                m = mx.re.search(r"\bName\s*:\s*([^;\n\r]+)", s, flags=mx.re.IGNORECASE)
                if not m:
                    return None
                name = m.group(1).strip()
                return name or None

            persona_name = _extract_name_from_persona_info(persona_info)

            static_meta = {
                "speaker_a": conv.get("speaker_a", "A"),
                "speaker_b": conv.get("speaker_b", "B"),
                "persona_info": persona_info,
                "persona_name": persona_name,
                "sample_id": sample_id,
            }

            session_keys = []
            for key in conv.keys():
                if not isinstance(key, str) or not key.startswith("session_"):
                    continue
                # keep only plain session bucket keys like session_1/session_2
                if not mx.re.fullmatch(r"session_\d+", key):
                    continue
                session_keys.append(key)
            session_keys = sorted(session_keys, key=lambda x: int(x.split("_")[1]))

            if mx.Config.LIMIT_SESSIONS is not None:
                session_keys = session_keys[: mx.Config.LIMIT_SESSIONS]

            current_session_id = None
            for k in mx._tqdm(session_keys, total=len(session_keys), desc=f"U{user_id} locomo sessions", leave=False):
                t_key = f"{k}_date_time"
                start_key = f"{k}_start_time"
                end_key = f"{k}_end_time"

                m_sid = mx.re.fullmatch(r"session_(\d+)", str(k))
                sid_num = m_sid.group(1) if m_sid else str(k).replace("_", "")
                locomo_session_id = f"c{conv_idx_1based}_session{sid_num}" if use_locomo_session_prefix else k

                session_start_time = conv.get(start_key, conv.get(t_key, "Unknown"))
                session_end_time = conv.get(end_key, session_start_time)

                if "json" in str(session_start_time):
                    session_start_time = "Unknown"
                if "json" in str(session_end_time):
                    session_end_time = session_start_time

                session_msgs = conv.get(k)
                if not isinstance(session_msgs, list):
                    if on_session_done is not None:
                        try:
                            on_session_done()
                        except Exception:
                            pass
                    continue

                current_session_id = locomo_session_id
                for idx, m in enumerate(session_msgs, start=1):
                    if not isinstance(m, dict):
                        continue

                    msg_time = m.get("time") or m.get("timestamp") or session_start_time
                    msg = {
                        "role": m.get("speaker"),
                        "text": str(m.get("text", "") or "").strip(),
                        "time": msg_time,
                        "dia_id": m.get("dia_id"),
                        "_temp_session_start_time": session_start_time,
                        "_temp_session_end_time": session_end_time,
                    }
                    self._process(msg, static_meta, user_id, current_session_id, idx)

                if on_session_done is not None:
                    try:
                        on_session_done()
                    except Exception:
                        pass

            self._seal(static_meta, user_id, current_session_id)
            all_samples_boxes.extend(self.boxes)

            if do_write_incremental:
                self.save_incremental(self.boxes, append=True)

        return all_samples_boxes

    def build_all_longmemeval(
        self,
        raw_list_override: List[Dict[str, Any]] | None = None,
        user_id_start: int = 0,
        write_incremental: bool | None = None,
        on_session_done=None,
    ):
        """Build blocks for haystack-style QA samples.

        Expected sample format (top-level list):
        {
            "question_id": "...",
            "question_type": "...",
            "question": "...",
            "question_date": "YYYY/MM/DD (Dow) HH:MM",
            "answer": "...",
            "answer_session_ids": [...],
            "haystack_dates": ["YYYY/MM/DD (Dow) HH:MM", ...],
            "haystack_session_ids": ["sid1", "sid2", ...],
            "haystack_sessions": [
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
                ...
            ]
        }

        Notes:
        - `haystack_sessions` is required for block construction.
        - Session date and session id are aligned by index with `haystack_sessions`.
        - If date/id is missing by index, falls back to `question_date` and generated `session_i`.
        """
        mx = _mx()
        all_samples_boxes: List[Dict[str, Any]] = []
        raw_list = raw_list_override if raw_list_override is not None else mx._load_raw_conversations(mx.Config.RAW_DATA_FILE)

        if mx.Config.LIMIT_CONVERSATIONS is not None:
            raw_list = raw_list[: mx.Config.LIMIT_CONVERSATIONS]

        if mx._is_main_thread():
            mx.logger.info("🏗️  [BUILD-HAYSTACK-QA] Processing %s Samples...", len(raw_list))

        do_write_incremental = mx.Config.CHECKPOINT_EVERY_SAMPLE if write_incremental is None else bool(write_incremental)

        for i, sample in enumerate(mx._tqdm(raw_list, total=len(raw_list), desc="BUILD haystack QA conversations")):
            user_id = user_id_start + i
            question_id = str((sample or {}).get("question_id", "") or "").strip()

            if mx._is_main_thread():
                display_id = question_id or user_id
                try:
                    user_id_map = getattr(mx._THREAD_CTX, "user_id_map", None)
                    if user_id_map and user_id in user_id_map:
                        display_id = user_id_map[user_id]
                except Exception:
                    pass
                mx.logger.info("   Building Haystack QA Sample %s...", display_id)

            self.cluster = TopicClusterManager(self.worker)
            self.boxes = []
            self.msgs = []

            question_date = str((sample or {}).get("question_date", "") or "").strip()
            haystack_dates = (sample or {}).get("haystack_dates", []) or []
            haystack_session_ids = (sample or {}).get("haystack_session_ids", []) or []
            haystack_sessions = (sample or {}).get("haystack_sessions", []) or []

            if not isinstance(haystack_dates, list):
                haystack_dates = []
            if not isinstance(haystack_session_ids, list):
                haystack_session_ids = []
            if not isinstance(haystack_sessions, list):
                if on_session_done is not None:
                    try:
                        on_session_done()
                    except Exception:
                        pass
                continue

            static_meta = {
                "speaker_a": "user",
                "speaker_b": "assistant",
                "persona_info": "",
                "persona_name": None,
                "question_id": question_id,
                "question_type": (sample or {}).get("question_type", ""),
            }

            session_indices = list(range(len(haystack_sessions)))
            if mx.Config.LIMIT_SESSIONS is not None:
                session_indices = session_indices[: mx.Config.LIMIT_SESSIONS]

            current_session_id = None
            for s_idx in mx._tqdm(session_indices, total=len(session_indices), desc=f"U{user_id} haystack sessions", leave=False):
                session_msgs = haystack_sessions[s_idx]
                if not isinstance(session_msgs, list):
                    if on_session_done is not None:
                        try:
                            on_session_done()
                        except Exception:
                            pass
                    continue

                session_id = str(haystack_session_ids[s_idx]).strip() if s_idx < len(haystack_session_ids) else ""
                if not session_id:
                    session_id = f"session_{s_idx + 1}"

                session_time = str(haystack_dates[s_idx]).strip() if s_idx < len(haystack_dates) else ""
                if not session_time:
                    session_time = question_date or "Unknown"

                current_session_id = session_id
                for idx, m in enumerate(session_msgs, start=1):
                    if not isinstance(m, dict):
                        continue

                    role = m.get("role")
                    text = str(m.get("content", m.get("text", "")) or "").strip()
                    msg_time = m.get("time") or m.get("timestamp") or session_time

                    msg = {
                        "role": role,
                        "text": text,
                        "time": msg_time,
                        "_temp_session_start_time": session_time,
                        "_temp_session_end_time": session_time,
                    }
                    self._process(msg, static_meta, user_id, current_session_id, idx)

                if on_session_done is not None:
                    try:
                        on_session_done()
                    except Exception:
                        pass

            self._seal(static_meta, user_id, current_session_id)
            all_samples_boxes.extend(self.boxes)

            if do_write_incremental:
                self.save_incremental(self.boxes, append=True)

        return all_samples_boxes


    def _process(self, msg, meta, user_id, session_id, idx=None):
        mx = _mx()
        msg["_temp_session_id"] = session_id
        if idx is not None:
            msg["_temp_idx"] = idx

        if not self.msgs:
            self.msgs.append(msg)
            return

        last_session = self.msgs[-1].get("_temp_session_id")
        if last_session != session_id:
            self._seal(meta, user_id, last_session)
            self.msgs.append(msg)
            return

        if len(self.msgs) == 1:
            self.msgs.append(msg)
            return

        window = min(len(self.msgs), max(1, mx.Config.BUILD_PREV_MSGS))
        prev_slice = self.msgs[-window:]
        prev_msgs = [f"{m['role']}: {m['text']}" for m in prev_slice]

        curr_msg_str = f"{msg['role']}: {msg['text']}"
        res = self.worker.check_relation(prev_msgs, curr_msg_str, note=f"U{user_id}_Overhead_Split")

        mx.TraceLogger.log(
            mx.Config.BUILD_TRACE_FILE,
            {
                "type": "split_check",
                "user_id": user_id,
                "session_id": session_id,
                "prev_msgs": prev_msgs,
                "curr_msg": curr_msg_str,
                "decision": res,
            },
        )

        if res == "Yes":
            self.msgs.append(msg)
        else:
            self._seal(meta, user_id, last_session)
            self.msgs.append(msg)

    def _seal(self, meta, user_id, session_id):
        mx = _mx()
        if not self.msgs:
            return
        if self.cluster is None:
            self.cluster = TopicClusterManager(self.worker)

        start_idx = self.msgs[0].get("_temp_idx", 1)
        end_idx = self.msgs[-1].get("_temp_idx", 1)

        content_to_save = []
        for m in self.msgs:
            m_copy = dict(m)
            m_copy.pop("_temp_session_id", None)
            m_copy.pop("_temp_idx", None)
            m_copy.pop("_temp_session_start_time", None)
            m_copy.pop("_temp_session_end_time", None)
            content_to_save.append(m_copy)

        session_start_time = self.msgs[0].get("_temp_session_start_time") or self.msgs[0].get("time")
        session_end_time = self.msgs[0].get("_temp_session_end_time") or self.msgs[-1].get("time") or session_start_time

        raw_box = {
            "user_id": user_id,
            "box_id": self.bid,
            "background_info": {"start_time": session_start_time, "end_time": session_end_time},
            "content": content_to_save,
            "coverage": {"session_id": session_id, "start_idx": start_idx, "end_idx": end_idx},
            # ✅ pass persona info/name into this box (from build_all -> static_meta)
            "persona_info": meta.get("persona_info", ""),
            "persona_name": meta.get("persona_name", None),
        }

        self.cluster.process_new_box(raw_box, user_id)

        content_text = self.cluster._get_content_str(raw_box)
        rt = raw_box.get("runtime_info", {}) or {}

        messages_count = len(content_to_save)
        content_tokens = self.worker.count_tokens(content_text)
        enrich_text = " ".join(
            [
                content_text,
                rt.get("topic", ""),
                rt.get("keywords", ""),
                mx._events_to_text(rt.get("events", [])),
            ]
        ).strip()
        enriched_tokens = self.worker.count_tokens(enrich_text)
        ratio = enriched_tokens / max(1, content_tokens)

        self.token_ratios.append(ratio)
        self.msg_counts.append(messages_count)
        self.box_token_pairs.append((content_tokens, enriched_tokens))

        session_start_iso = mx.to_iso8601_if_possible(str(session_start_time or "Unknown"))
        session_end_iso = mx.to_iso8601_if_possible(str(session_end_time or session_start_time or "Unknown"))
        observed_time = session_end_iso if session_end_iso != "Unknown" else session_start_iso

        # ✅ choose persona_name (priority: meta > raw_box > runtime_info)
        persona_name = meta.get("persona_name") or raw_box.get("persona_name") or rt.get("persona_name")

        def _apply_person_name(desc: str, name: str | None) -> str:
            if not name:
                return (desc or "").strip()
            d = (desc or "").strip()
            if d.startswith("User's "):
                return f"{name}'s " + d[len("User's ") :]
            if d.startswith("User "):
                return f"{name} " + d[len("User ") :]
            if d == "User":
                return name
            return d

        # helper: detect explicit date in description and convert to ISO-like
        def _extract_explicit_date_iso(desc: str) -> str | None:
            s = (desc or "").strip()
            if not s:
                return None
            # ISO date
            m = mx.re.search(r"\b\d{4}-\d{2}-\d{2}\b", s)
            if m:
                return m.group(0)
            # Dataset-like date: "Sep 04, 2025" (allow any case)
            m2 = mx.re.search(r"\b[A-Za-z]{3} \d{2}, \d{4}\b", s)
            if m2:
                iso = mx.to_iso8601_if_possible(m2.group(0))
                return iso.split("T", 1)[0] if iso and iso != "Unknown" else None
            return None

        # 2/3) time_metadata rule:
        # Only emit startTime/endTime if the DESCRIPTION contains an explicit date.
        events_out: List[Dict[str, Any]] = []
        for ei, ev in enumerate(rt.get("events", []) or []):

            llm_type = ""
            if isinstance(ev, dict):
                ev_start_raw = ev.get("start_time", "Unknown")
                ev_end_raw = ev.get("end_time", ev.get("start_time", "Unknown"))
                desc = str(ev.get("description", "") or "").strip()
                # ✅ preserve LLM temporal type (OCCURRENCE/STATE/ATTRIBUTE/INTENTION)
                llm_type = str(ev.get("type", "") or "").strip().upper()
            else:
                line = mx._event_any_to_line(ev)
                obj = (
                    mx._event_line_to_obj(line)
                    if line
                    else {"start_time": "Unknown", "end_time": "Unknown", "description": str(ev)}
                )
                ev_start_raw = obj.get("start_time", "Unknown")
                ev_end_raw = obj.get("end_time", obj.get("start_time", "Unknown"))
                desc = str(obj.get("description", "") or "").strip()

            # ✅ inject name into event description (output-level fallback)
            desc = _apply_person_name(desc, persona_name)

            # ❌ REMOVED: Event label classification (expensive LLM calls)
            # This was only used to generate categories field, which is unused in retrieval
            # labels = self.worker.classify_event_labels(
            #     desc,
            #     top_k=getattr(mx.Config, "EVENT_LABEL_TOP_K", 3),
            #     with_meta=True,
            # )
            # filtered_labels: List[Dict[str, Any]] = []
            # for it in (labels or []):
            #     if isinstance(it, dict):
            #         lab = str(it.get("label") or "").strip()
            #         if not lab:
            #             continue
            #         try:
            #             conf = float(it.get("confidence", 0.0) or 0.0)
            #         except Exception:
            #             conf = 0.0
            #         if conf >= 0.7:
            #             filtered_labels.append({"label": lab, "confidence": conf})
            #     elif isinstance(it, str):
            #         lab = it.strip()
            #         if lab:
            #             filtered_labels.append({"label": lab, "confidence": 1.0})

            # Set labels to empty list (field removed from schema)
            filtered_labels = []

            # default: observedTime only
            time_md: Dict[str, Any] = {"observedTime": observed_time}

            # Try to extract date from description first
            explicit_date = _extract_explicit_date_iso(desc)
            if explicit_date:
                time_md["startTime"] = explicit_date
                time_md["endTime"] = explicit_date
                time_md["granularity"] = "day"
            # If no date in description, use event's start_time/end_time if available
            elif ev_start_raw and str(ev_start_raw).strip() not in ("Unknown", ""):
                # Convert to ISO format if needed
                start_iso = mx.to_iso8601_if_possible(str(ev_start_raw))
                end_iso = mx.to_iso8601_if_possible(str(ev_end_raw)) if ev_end_raw else start_iso
                if start_iso != "Unknown":
                    # Extract date part only (YYYY-MM-DD)
                    time_md["startTime"] = start_iso.split("T")[0] if "T" in start_iso else start_iso
                    time_md["endTime"] = end_iso.split("T")[0] if "T" in end_iso else end_iso
                    time_md["granularity"] = "day"

            # ✅ Prefer LLM classification; fallback to heuristic only if missing/invalid
            if llm_type in {"OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"}:
                event_temporal_type = llm_type
            else:
                ht = str(mx.infer_event_type(desc) or "").strip().upper()
                # keep old heuristic compatibility: EVENT -> OCCURRENCE; otherwise default to ATTRIBUTE
                if ht == "EVENT":
                    event_temporal_type = "OCCURRENCE"
                elif ht in {"OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"}:
                    event_temporal_type = ht
                else:
                    event_temporal_type = "ATTRIBUTE"

            events_out.append(
                {
                    "event_id": f"{user_id}_{self.bid}_e{ei}",
                    "description": desc,
                    "event_temporal_type": event_temporal_type,
                    "time_metadata": time_md,
                    # ❌ REMOVED: "labels": filtered_labels,
                }
            )

        # ❌ REMOVED: Category aggregation from labels (topic filtering is disabled)
        # categories_set = set()
        # for e in events_out:
        #     for lab in (e.get("labels") or []):
        #         if isinstance(lab, dict):
        #             name = str(lab.get("label") or "").strip()
        #             if name:
        #                 categories_set.add(name)
        #         elif isinstance(lab, str):
        #             if lab.strip():
        #                 categories_set.add(lab.strip())
        # categories = sorted(categories_set)

        # Set categories to empty list (field removed from schema)
        categories = []

        # block_event_start/end_time: (keep your existing logic unchanged)
        block_start_dt = None
        block_end_dt = None
        for e in events_out:
            s = e.get("time_metadata", {}).get("startTime", "Unknown")
            t = e.get("time_metadata", {}).get("endTime", "Unknown")

            try:
                sdt, _ = mx._parse_event_time_range(f"[{s}, {s}] x") if s != "Unknown" else (None, None)
            except Exception:
                sdt = None
            try:
                edt, _ = mx._parse_event_time_range(f"[{t}, {t}] x") if t != "Unknown" else (None, None)
            except Exception:
                edt = None

            if sdt is not None:
                block_start_dt = sdt if block_start_dt is None else min(block_start_dt, sdt)
            if edt is not None:
                block_end_dt = edt if block_end_dt is None else max(block_end_dt, edt)

        if block_start_dt is None:
            try:
                sdt, _ = mx._parse_event_time_range(f"[{session_start_iso}, {session_start_iso}] x")
                block_start_dt = sdt
            except Exception:
                block_start_dt = None
        if block_end_dt is None:
            try:
                edt, _ = mx._parse_event_time_range(f"[{session_end_iso}, {session_end_iso}] x")
                block_end_dt = edt
            except Exception:
                block_end_dt = None

        block_event_start_time = block_start_dt.strftime("%Y-%m-%d") if block_start_dt else (
            session_start_iso.split("T", 1)[0] if session_start_iso != "Unknown" else "Unknown"
        )
        block_event_end_time = block_end_dt.strftime("%Y-%m-%d") if block_end_dt else (
            session_end_iso.split("T", 1)[0] if session_end_iso != "Unknown" else "Unknown"
        )

        final_box = {
            "user_id": user_id,
            "block_id": self.bid,
            "coverage": raw_box.get("coverage", {}),
            "temporal_index": {
                "sessionstart_time": session_start_iso,
                "sessionend_time": session_end_iso,
                "block_event_start_time": block_event_start_time,
                "block_event_end_time": block_event_end_time,
            },
            "features": {
                "content_text": content_text,  # ✅ ADDED: Original conversation text for generation
                "topic_kw_text": rt.get("topic_kw_text", ""),
            },
            "events_count": len(events_out),
            "events": events_out,
        }

        # Graph upsert: one Event node per event, plus a Block node containing them.
        if self.event_graph is not None:
            try:
                store = mx.EmbeddingStore(self.worker, user_id)
                event_vec_cache = _load_event_vector_cache(store)
                for ev in events_out:
                    payload = build_graph_payload_from_memblock(block=final_box, event=ev, user_id=str(user_id))
                    payload["created_at"] = payload.get("created_at") or mx.datetime.utcnow().isoformat() + "Z"
                    memory_id = str(ev.get("event_id") or "")
                    if memory_id:
                        self.event_graph.upsert_event_node(memory_id=memory_id, payload=payload)
                        if self.entity_extractor is not None:
                            extract_and_store_event_entities(
                                event_graph=self.event_graph,
                                extractor=self.entity_extractor,
                                event_id=memory_id,
                                user_id=str(user_id),
                                event_description=str(ev.get("description") or ""),
                            )
                        desc = str(ev.get("description") or "").strip()
                        if desc:
                            vec = store.get_vector(
                                f"graph_event_{memory_id}",
                                "event",
                                desc,
                                note=f"U{user_id}_E{memory_id}_graph",
                            )
                            if vec:
                                threshold = float(getattr(self.graph_config, "graph_similarity_threshold", 0.8) or 0.8)
                                limit = int(getattr(self.graph_config, "graph_similar_limit", 20) or 20)
                                _link_similar_events_global(
                                    graph=self.event_graph,
                                    user_id=str(user_id),
                                    event_id=memory_id,
                                    event_vec=vec,
                                    existing_event_vecs=event_vec_cache,
                                    threshold=threshold,
                                    limit=limit,
                                )
                                event_vec_cache[memory_id] = vec

                store.flush()
            except Exception:
                # Best-effort: graph failures should not break build.
                pass

        self.bid += 1
        self.boxes.append(final_box)
        self.total_boxes += 1
        self.msgs = []

    def _write_boxes(self, boxes):
        mx = _mx()
        mode = "a" if mx.os.path.exists(mx.Config.FINAL_CONTENT_FILE) else "w"
        mx.os.makedirs(mx.os.path.dirname(mx.Config.FINAL_CONTENT_FILE), exist_ok=True)
        with mx._APPEND_LOCK:
            with open(mx.Config.FINAL_CONTENT_FILE, mode, encoding="utf-8") as f:
                for b in boxes:
                    f.write(mx.json.dumps(b, ensure_ascii=False) + "\n")

    def save(self, boxes):
        mx = _mx()
        self._write_boxes(boxes)
        mx.logger.info("✅ [BUILD] Complete. Saved %s boxes (appended).", len(boxes))

    def save_incremental(self, boxes, append=True):
        mx = _mx()
        if not boxes:
            return
        self._write_boxes(boxes)
        mx.logger.info("✅ [BUILD] Checkpoint saved: +%s boxes (appended)", len(boxes))

    def summarize_and_log(self):
        mx = _mx()
        total_boxes = self.total_boxes or len(self.token_ratios)
        avg_msg = sum(self.msg_counts) / total_boxes if total_boxes else 0
        avg_ratio = sum(self.token_ratios) / total_boxes if total_boxes else 0
        total_content_tokens = sum(p[0] for p in self.box_token_pairs)
        total_enriched_tokens = sum(p[1] for p in self.box_token_pairs)

        llm_stats = mx.TokenAnalyzer.get_stage_stats("build")
        total_messages = sum(self.msg_counts)
        boxes_denom = max(total_boxes, 1)
        msgs_denom = max(total_messages, 1)
        llm_calls = llm_stats.get("calls", 0)

        summary = {
            "run_id": mx.Config.RUN_ID,
            "timestamp": mx.datetime.now().isoformat(),
            "boxes": total_boxes,
            "total_messages": total_messages,
            "avg_messages_per_box": round(avg_msg, 3),
            "avg_token_ratio_enriched_over_content": round(avg_ratio, 3),
            "total_content_tokens": total_content_tokens,
            "total_enriched_tokens": total_enriched_tokens,
            "llm_calls_build": llm_stats.get("calls", 0),
            "llm_prompt_tokens": llm_stats.get("prompt", 0),
            "llm_completion_tokens": llm_stats.get("completion", 0),
            "llm_total_tokens": llm_stats.get("total", 0),
            "avg_llm_calls_per_box": round(llm_calls / boxes_denom, 3),
            "avg_llm_calls_per_message": round(llm_calls / msgs_denom, 3),
            "avg_prompt_tokens_per_box": round(llm_stats.get("prompt", 0) / boxes_denom, 3),
            "avg_completion_tokens_per_box": round(llm_stats.get("completion", 0) / boxes_denom, 3),
            "avg_total_tokens_per_box": round(llm_stats.get("total", 0) / boxes_denom, 3),
            "avg_prompt_tokens_per_message": round(llm_stats.get("prompt", 0) / msgs_denom, 3),
            "avg_completion_tokens_per_message": round(llm_stats.get("completion", 0) / msgs_denom, 3),
            "avg_total_tokens_per_message": round(llm_stats.get("total", 0) / msgs_denom, 3),
        }

        mx.os.makedirs(mx.os.path.dirname(mx.Config.BUILD_STATS_FILE), exist_ok=True)
        with mx._APPEND_LOCK:
            with open(mx.Config.BUILD_STATS_FILE, "a", encoding="utf-8") as f:
                f.write(mx.json.dumps(summary, ensure_ascii=False) + "\n")

        mx.logger.info(
            "ℹ️ Build stats | boxes=%s msgs=%s avg_msg=%.2f avg_ratio=%.3f llm_calls=%s (per_box=%.3f per_msg=%.3f) tokens(prompt/out/total)=(%s/%s/%s)",
            total_boxes,
            total_messages,
            avg_msg,
            avg_ratio,
            llm_calls,
            llm_calls / boxes_denom,
            llm_calls / msgs_denom,
            llm_stats.get("prompt", 0),
            llm_stats.get("completion", 0),
            llm_stats.get("total", 0),
        )
