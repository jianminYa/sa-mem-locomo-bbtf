"""
Enhanced retrieval with query parsing, time-based filtering, and topic-based filtering.
Integrates query_pasing_byllm.py for intent recognition and metadata extraction.
"""
from __future__ import annotations

import csv
import json
import os
from time import time
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from query_pasing_byllm import QueryParser, SearchDirective, dispatch_temporal_filter
from interval_tree_index import TemporalIndex
from anchor_resolver import AnchorResolver

try:
    from graph_storage import GraphConfig, MemgraphEventGraph
except Exception:  # pragma: no cover - optional dependency
    GraphConfig = None  # type: ignore
    MemgraphEventGraph = None  # type: ignore


def _mx():
    """Lazy import to avoid circular imports."""
    import memblock_extractor as mx
    return mx


class EnhancedRetriever:
    """
    Enhanced retrieval system with:
    1. Query parsing for time and topic extraction
    2. Temporal filtering using interval trees
    3. Vector similarity ranking
    """

    def __init__(
        self,
        worker: Any,
        top_k: int = None,
        graph_expand: bool = False,
        graph_min_score: float = 0.7,
        graph_limit: int = 200,
        graph_include_relations: bool = True,
        use_anchor: bool = True,
        axis_mode: str = "auto",
    ):
        mx = _mx()
        self.worker = worker
        self.top_k = mx.Config.TOP_K_RETRIEVE if top_k is None else top_k

        # Bi-temporal ablation switch (mirrors LoCoMo retriever).
        # - "auto"    : use _infer_time_axis result from QueryParser (default).
        # - "session" : force axis=SESSION (only query session_tree).
        #               ANCHOR queries degrade to full pool.
        # - "event"   : force axis=EVENT (only query event_tree).
        # - "none"    : skip the entire temporal filtering block.
        axis_mode = (axis_mode or "auto").lower()
        if axis_mode not in {"auto", "session", "event", "none"}:
            raise ValueError(f"Invalid axis_mode={axis_mode!r}; must be auto/session/event/none.")
        self.axis_mode = axis_mode
        self.use_anchor = bool(use_anchor)

        # Query parser for intent and time extraction
        self.query_parser = QueryParser(
            api_key=mx.Config.API_KEY,
            base_url=mx.Config.BASE_URL,
            model=mx.Config.LLM_MODEL
        )

        # Data structures
        self.all_blocks: List[Dict[str, Any]] = []
        self.temporal_index: TemporalIndex = TemporalIndex()
        self.blocks_by_user: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        self.trace_map: Dict[Any, Dict[int, List[int]]] = {}
        self.anchor_resolver: Optional[AnchorResolver] = None
        # EmbeddingStore lifecycle: one store per user during one run
        self.store_by_user: Dict[Any, Any] = {}
        self.query_count_by_user: Dict[Any, int] = defaultdict(int)

        # Graph expansion config (optional, no impact on core retrieval)
        self.graph_expand = bool(graph_expand)
        self.graph_min_score = float(graph_min_score)
        self.graph_limit = int(graph_limit)
        self.graph_include_relations = bool(graph_include_relations)
        self.graph: Optional[Any] = None

        # Flush policy:
        # 0 or None means disable periodic flush, only flush at user end / run end
        self.flush_every_n_queries: int = 20

    def load(self):
        """Load memory blocks and build indices."""
        mx = _mx()

        # Load blocks
        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            self.all_blocks = [json.loads(l) for l in f if l.strip()]

        # Group by user
        for block in self.all_blocks:
            user_id = block.get("user_id")
            if user_id is not None:
                self.blocks_by_user[user_id].append(block)

        # Build temporal index
        self.temporal_index.build_from_blocks(self.all_blocks)

        # Initialize anchor resolver (skip when --no-use-anchor is requested).
        if self.use_anchor:
            self.anchor_resolver = AnchorResolver(self.temporal_index, self.worker)
        else:
            self.anchor_resolver = None

        # Load traces (for compatibility)
        self.trace_map = self._load_traces(mx.Config.TIME_TRACE_FILE)

        # Optional graph init for expansion
        if self.graph_expand and MemgraphEventGraph is not None and GraphConfig is not None:
            try:
                cfg = GraphConfig()
                self.graph = MemgraphEventGraph(
                    url=cfg.memgraph_url,
                    username=cfg.memgraph_username,
                    password=cfg.memgraph_password,
                )
            except Exception as e:
                mx.logger.warning("⚠️ Graph expansion disabled: %s", e)
                self.graph = None

        mx.logger.info("✅ Loaded %d blocks, built temporal index", len(self.all_blocks))

    def _expand_graph_for_blocks(self, user_id: Any, block_ids: List[Any]) -> Dict[str, Any]:
        mx = _mx()
        if not self.graph or not block_ids:
            return {}

        try:
            events_by_block = self.graph.get_events_by_block_ids(
                user_id=str(user_id),
                block_ids=block_ids,
            )
            seed_event_ids: List[str] = []
            seen_seed = set()
            for _, items in (events_by_block or {}).items():
                for ev in items:
                    eid = ev.get("event_id")
                    if not eid:
                        continue
                    eid_str = str(eid)
                    if eid_str not in seen_seed:
                        seed_event_ids.append(eid_str)
                        seen_seed.add(eid_str)

            similar_edges = self.graph.get_similar_events_1hop(
                user_id=str(user_id),
                event_ids=seed_event_ids,
                min_score=self.graph_min_score,
                limit=self.graph_limit,
            )

            seed_set = set(seed_event_ids)
            edge_map: Dict[tuple, Dict[str, Any]] = {}
            neighbor_ids: set[str] = set()
            for edge in similar_edges or []:
                a = str(edge.get("from_event_id") or "")
                b = str(edge.get("to_event_id") or "")
                if not a or not b:
                    continue
                left, right = (a, b) if a <= b else (b, a)
                key = (left, right)
                score = edge.get("score")
                prev = edge_map.get(key)
                if prev is None or (score is not None and score > prev.get("score", -1.0)):
                    edge_map[key] = {
                        "from_event_id": a,
                        "to_event_id": b,
                        "score": score,
                    }
                if a in seed_set and b not in seed_set:
                    neighbor_ids.add(b)
                if b in seed_set and a not in seed_set:
                    neighbor_ids.add(a)

            expanded_event_ids = [eid for eid in neighbor_ids if eid not in seed_set]
            expanded_events = self.graph.get_events_by_ids(
                user_id=str(user_id),
                event_ids=expanded_event_ids,
            )

            relations: List[Dict[str, Any]] = []
            if self.graph_include_relations:
                relation_event_ids = list(seed_set | set(expanded_event_ids))
                relations = self.graph.get_relations_by_event_ids(
                    user_id=str(user_id),
                    event_ids=relation_event_ids,
                    limit=self.graph_limit,
                )

            rel_seen = set()
            rel_out: List[Dict[str, Any]] = []
            for rel in relations or []:
                key = (
                    str(rel.get("source") or ""),
                    str(rel.get("relationship") or ""),
                    str(rel.get("destination") or ""),
                    str(rel.get("event_id") or ""),
                )
                if key in rel_seen:
                    continue
                rel_seen.add(key)
                rel_out.append(rel)

            return {
                "graph_enabled": True,
                "seed_event_ids": seed_event_ids,
                "expanded_events": expanded_events,
                "similar_events": list(edge_map.values()),
                "relations": rel_out,
            }
        except Exception as e:
            mx.logger.warning("⚠️ Graph expansion failed: %s", e)
            return {"graph_enabled": False, "error": str(e)}

    def _get_or_create_store(self, user_id: Any):
        mx = _mx()
        if user_id in self.store_by_user:
            store = self.store_by_user[user_id]
            mx.logger.info(
                "🔍 VectorStore reuse user_id=%s path=%s file_exists=%s loaded_keys=%d",
                str(user_id),
                getattr(store, "path", ""),
                os.path.exists(getattr(store, "path", "")),
                len(getattr(store, "data", {}) or {}),
            )
            return store, 0.0

        import time
        t0 = time.perf_counter()
        store = mx.EmbeddingStore(self.worker, user_id)
        t_store_init = time.perf_counter() - t0
        self.store_by_user[user_id] = store
        mx.logger.info(
            "🔍 VectorStore load user_id=%s path=%s file_exists=%s loaded_keys=%d load_sec=%.6f",
            str(user_id),
            store.path,
            os.path.exists(store.path),
            len(store.data),
            t_store_init,
        )
        return store, t_store_init

    def _flush_user_store(self, user_id: Any, force: bool = False) -> float:
        import time
        store = self.store_by_user.get(user_id)
        if store is None:
            return 0.0
        if (not force) and (not getattr(store, "dirty", False)):
            return 0.0

        t0 = time.perf_counter()
        store.flush()
        return time.perf_counter() - t0

    def _flush_all_stores(self):
        mx = _mx()
        flushed = 0
        for uid in list(self.store_by_user.keys()):
            try:
                dt = self._flush_user_store(uid, force=True)
                if dt > 0:
                    flushed += 1
            except Exception as e:
                mx.logger.warning("⚠️ flush failed for user %s: %s", uid, e)
        mx.logger.info("ℹ️ Final flush completed for %d user stores", flushed)

    @staticmethod
    def _load_traces(path: str) -> Dict[Any, Dict[int, List[int]]]:
        """Load trace mapping from file."""
        if not os.path.exists(path):
            return {}

        trace_map: Dict[Any, Dict[int, List[int]]] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    uid = t.get("user_id")
                    ids = t.get("box_ids", []) or []
                    if uid is None:
                        continue
                    trace_map.setdefault(uid, {})
                    for bid in ids:
                        trace_map[uid][int(bid)] = [int(x) for x in ids]
        except Exception:
            return {}
        return trace_map

    def _filter_by_metadata(
        self,
        pool: List[Dict[str, Any]],
        directive: SearchDirective
    ) -> List[Dict[str, Any]]:
        """
        Filter memory blocks based on parsed query metadata.

        Args:
            pool: Initial pool of candidate blocks
            directive: Parsed query directive with time and topic constraints

        Returns:
            Filtered list of blocks
        """
        mx = _mx()
        import time

        # Timing variables
        t_temporal = 0.0
        t_anchor = 0.0
        t_event_type = 0.0

        # Use block index in all_blocks as unique identifier instead of block_id
        # because block_id is not unique across sessions
        block_to_idx = {id(b): i for i, b in enumerate(self.all_blocks)}
        pool_ids = {block_to_idx.get(id(b), i) for i, b in enumerate(pool)}

        # DEBUG: Log pool_ids size
        mx.logger.info("🔍 pool_ids size = %d blocks", len(pool_ids))

        # -------------------------
        # 1) Temporal filtering
        # -------------------------
        t_temporal_start = time.perf_counter()
        time_filtered_ids = pool_ids

        # DEBUG: Log temporal filtering decision
        mx.logger.info("🔍 use_interval_tree=%s, time_constraint.type=%s",
                      directive.use_interval_tree, directive.time_constraint.type)

        def _temporal_query_ids(
            start_date: Optional[str],
            end_date: Optional[str],
            time_type: str,
            use_event_time: bool,
        ) -> set:
            ids = set(self.temporal_index.query_temporal(
                start_date=start_date,
                end_date=end_date,
                time_type=time_type,
                use_event_time=use_event_time,
            ))
            return ids & pool_ids

        # Log intent-based filtering decision
        if directive.intent in {"PLANNING", "STATIC"}:
            mx.logger.info(
                "🔮 %s query detected. Skipping temporal filtering (use_interval_tree=%s).",
                directive.intent, directive.use_interval_tree
            )

        if directive.use_interval_tree and directive.time_constraint.type != "NONE":
            tc = directive.time_constraint
            axis = (getattr(directive, "time_axis", "BOTH_UNION") or "BOTH_UNION").upper()

            # ====== Bi-temporal ablation override ======
            if self.axis_mode == "none":
                mx.logger.info(
                    "🧪 [axis_mode=none] Skipping temporal filtering for tc.type=%s (axis=%s).",
                    tc.type, axis,
                )
                time_filtered_ids = pool_ids
            elif tc.type == "ANCHOR":
                if self.axis_mode == "session":
                    mx.logger.info(
                        "🧪 [axis_mode=session] ANCHOR query → degrade to full pool "
                        "(session_tree has no anchor semantics)."
                    )
                    time_filtered_ids = pool_ids
                elif not self.use_anchor:
                    mx.logger.info("🔍 ANCHOR query detected but anchor resolution is disabled → using full pool")
                    time_filtered_ids = pool_ids
                else:
                    mx.logger.info("🔍 ANCHOR query detected: '%s' %s", tc.anchor_event, tc.anchor_relation)

                    if self.anchor_resolver:
                        t_anchor_start = time.perf_counter()
                        anchor_start, anchor_end, anchor_type = self.anchor_resolver.resolve_anchor(
                            tc.anchor_event or tc.raw_text or "",
                            tc.anchor_relation or "DURING",
                            pool[0].get("user_id") if pool else None,
                            pool
                        )
                        t_anchor = time.perf_counter() - t_anchor_start

                        if anchor_type != "NONE" and anchor_start:
                            # Anchors are real-world events -> event-time only.
                            ids_evt = _temporal_query_ids(
                                anchor_start.isoformat(),
                                anchor_end.isoformat() if anchor_end else None,
                                anchor_type,
                                True,
                            )
                            mx.logger.info(
                                "🔍 Anchor temporal(event) type=%s [%s, %s] -> %d/%d blocks",
                                anchor_type, anchor_start, anchor_end, len(ids_evt), len(pool_ids)
                            )
                            time_filtered_ids = ids_evt
                        else:
                            mx.logger.warning("⚠️ Anchor resolution failed, using full pool")
                            time_filtered_ids = pool_ids
                    else:
                        time_filtered_ids = pool_ids

            else:
                # Non-anchor temporal: axis-driven dispatcher replaces the
                # old "try event, fall back to session" black/white logic.
                if self.axis_mode == "session":
                    axis = "SESSION"
                elif self.axis_mode == "event":
                    axis = "EVENT"
                # axis_mode == "auto" keeps the QueryParser-inferred axis.
                time_filtered_ids, mode_used = dispatch_temporal_filter(
                    axis,
                    query_event=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, True),
                    query_session=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, False),
                )
                mx.logger.info(
                    "🔍 Temporal filter (axis=%s, mode=%s, axis_mode=%s): %s [%s, %s] -> %d/%d blocks",
                    axis, mode_used, self.axis_mode, tc.type, tc.start, tc.end,
                    len(time_filtered_ids), len(pool_ids)
                )

        # DEBUG: Log time filtering result
        mx.logger.info("🔍 After temporal filtering, time_filtered_ids size = %d blocks", len(time_filtered_ids))
        t_temporal = time.perf_counter() - t_temporal_start
    
        # -------------------------
        # 2) Topic filtering - DISABLED (removed per user request)
        # -------------------------
        # 直接跳过主题过滤，使用时间过滤的结果
    

        # -------------------------
        # 3) Event type filtering - DISABLED (removed per user request)
        # -------------------------
        # 直接跳过事件类型过滤，使用类别过滤的结果
      


        # Final filtering result
        final_count = len(time_filtered_ids)
        if final_count == 0:
            # Safety fallback: hard metadata filters (temporal / type / category)
            # produced an empty set — usually because the query's time window
            # lies outside the memory's coverage. Returning [] would make the
            # enhanced retriever strictly worse than baseline on those queries.
            # Drop the hard filters and let vector ranking work on the full
            # candidate pool (same policy as baseline).
            mx.logger.warning(
                "⚠️ Filtering resulted in 0 blocks -> dropping hard filters and "
                "falling back to full pool of %d blocks for vector ranking.",
                len(pool_ids),
            )
            time_filtered_ids = pool_ids
            final_count = len(time_filtered_ids)
        elif final_count < 5:
            mx.logger.info(
                "✅ Filtering resulted in %d blocks (small but precise result)",
                final_count
            )
        else:
            mx.logger.info(
                "✅ Filtering resulted in %d blocks",
                final_count
            )

        # Log timing breakdown for metadata filtering
        mx.logger.info(
            "⏱️ Filter timing: temporal=%.6fs",
            t_temporal
        )

        # return [b for b in pool if mx._get_block_id(b) in time_filtered_ids]
        return [b for b in pool if block_to_idx.get(id(b)) in time_filtered_ids]


    def _score_and_rank(
        self,
        user_id: Any,
        qa: Dict[str, Any],
        use_enhanced: bool = True,
        qa_idx: int | None = None
    ) -> Tuple[Dict[str, List[int]], Dict[int, float], List[int], float]:
        """
        Score and rank blocks for a query with optional metadata filtering.

        Args:
            user_id: User identifier
            qa: Question-answer pair
            use_enhanced: If True, use query parsing and metadata filtering
            qa_idx: Index of the question-answer pair for logging purposes
        Returns:
            Tuple of (rankings, similarity_map, target_boxes)
        """
        mx = _mx()

        pool = self.blocks_by_user.get(user_id, [])
        if not pool:
            return {}, {}, [], 0.0

        mx.logger.info("🔍  Initial pool size = %d blocks", len(pool))

        question = qa.get("question", "") or ""
        q_id = qa.get("id", qa.get("question", ""))

        # Timing variables
        import time
        t_parse = 0.0
        t_filter = 0.0
        t_rank = 0.0
        # A/B/C/D profiling variables
        t_store_init = 0.0      # B: cache file load (EmbeddingStore init)
        t_query_vec = 0.0       # A: query embedding
        t_loop = 0.0            # D-main: block loop (get vec + cosine)
        t_sort = 0.0            # D-tail: sorting
        t_flush = 0.0           # C: cache flush
        block_vec_hit = 0
        block_vec_miss = 0
        pool_size_before = len(pool)
        pool_size_after = 0
        qvec_cache_hit = False
        # Parse query if enhanced mode is enabled
        directive = None
        if use_enhanced and question:
            try:
                t_parse_start = time.perf_counter()
                directive = self.query_parser.parse(question)
                t_parse = time.perf_counter() - t_parse_start
                mx.logger.info(
                    "📝 Query parsed: intent=%s, time=%s, source=%s",
                    directive.intent, directive.time_constraint.type, directive.parse_source
                )
            except Exception as e:
                mx.logger.warning("⚠️ Query parsing failed: %s", e)
                directive = None

        # Apply metadata filtering if directive is available
        if directive and use_enhanced:
            t_filter_start = time.perf_counter()
            filtered_pool = self._filter_by_metadata(pool, directive)
            t_filter = time.perf_counter() - t_filter_start
            mx.logger.info(
                "🔍 Metadata filtering: %d/%d blocks after filtering",
                len(filtered_pool), len(pool)
            )
        else:
            filtered_pool = pool

        pool_size_after = len(filtered_pool)
        # Vector similarity ranking on filtered pool
        

        # B) EmbeddingStore init (includes cache JSON load)

        # store = mx.EmbeddingStore(self.worker, user_id)
        store, t_store_init = self._get_or_create_store(user_id)
        t_rank_start = time.perf_counter()
        
        store_key_count_before = len(store.data)
        vector_file_size_mb = 0.0
        try:
            if os.path.exists(store.path):
                vector_file_size_mb = os.path.getsize(store.path) / (1024.0 * 1024.0)
        except Exception:
            vector_file_size_mb = 0.0
        mx.logger.info(
            "🔍 VectorStore inspect user_id=%s path=%s file_exists=%s file_mb=%.2f loaded_keys=%d init_sec=%.6f",
            str(user_id),
            store.path,
            os.path.exists(store.path),
            vector_file_size_mb,
            store_key_count_before,
            t_store_init,
        )
        # Use rewritten query if available, otherwise use original
        query_text = directive.rewritten_query if directive else question

        # A) Query embedding time + query cache hit
        q_key = f"qa_{user_id}_{q_id}"
        qvec_cache_hit = (q_key in store.data and "question" in store.data.get(q_key, {}))
        t_query_vec_start = time.perf_counter()
        qvec = store.get_vector(
            q_key,
            "question",
            query_text,
            note=f"U{user_id}_QA_Enhanced"
        )
        t_query_vec = time.perf_counter() - t_query_vec_start
        mx.logger.info(
            "⏱️ Query embedding time user_id=%s q_id=%s qvec_hit=%s qvec_sec=%.6f",
            str(user_id),
            str(q_id),
            str(qvec_cache_hit),
            t_query_vec,
        )
        
        mx.logger.info("🔄 Computing similarity for %d blocks...", len(filtered_pool))
        sim_map: Dict[int, float] = {}

        # D-main) Loop time (block vec get + cosine)
        t_loop_start = time.perf_counter()
        for _, b in enumerate(filtered_pool):
            bid = mx._get_block_id(b)
            key = f"{user_id}_{bid}"

            # Cache hit/miss for block vector
            if key in store.data and "content_event_topic_kw" in store.data.get(key, {}):
                block_vec_hit += 1
            else:
                block_vec_miss += 1
            # Build enriched text for embedding
            features = b.get("features", {})
            topic_kw = features.get("topic_kw_text", "")

            # Include event descriptions
            events = b.get("events", [])
            event_texts = [e.get("description", "") for e in events if e.get("description")]
            event_str = " | ".join(event_texts[:20])  # Limit to first 20 events

            text = f"{topic_kw} {event_str}".strip()

            v = store.get_vector(
                key,
                "content_event_topic_kw",
                text,
                note=f"U{user_id}_B{bid}_enhanced",
            )

            try:
                s = cosine_similarity([qvec], [v])[0][0] if v else -1.0
            except Exception:
                s = -1.0

            sim_map[bid] = float(s)
        t_loop = time.perf_counter() - t_loop_start
        mx.logger.info("✅ Similarity computation complete for %d blocks", len(filtered_pool))

        # Rank by similarity D)
        t_sort_start = time.perf_counter()
        ranked = [bid for bid, _ in sorted(sim_map.items(), key=lambda x: x[1], reverse=True)]
        t_sort = time.perf_counter() - t_sort_start
        rankings = {"content_event_topic_kw": ranked}

        # dirty_before_flush = bool(getattr(store, "dirty", False))
        # t_flush_start = time.perf_counter()
        # store.flush()
        # t_flush = time.perf_counter() - t_flush_start
        dirty_before_flush = bool(getattr(store, "dirty", False))
        self.query_count_by_user[user_id] += 1

        # periodic flush
        t_flush = 0.0
        do_periodic_flush = (
            self.flush_every_n_queries is not None
            and self.flush_every_n_queries > 0
            and (self.query_count_by_user[user_id] % self.flush_every_n_queries == 0)
        )
        if do_periodic_flush:
            t_flush = self._flush_user_store(user_id, force=False)

        store_key_count_after = len(store.data)
        
        t_rank = time.perf_counter() - t_rank_start

        # Log timing breakdown
        t_total = t_parse + t_filter + t_rank
        mx.logger.info(
            "⏱️ [Time Filtering] Timing: q_idx=%s parse=%.6fs, filter=%.6fs, rank=%.6fs, total=%.3fs, pool=%d",
            str(qa_idx) if qa_idx is not None else "NA",
            t_parse, t_filter, t_rank, t_total,
            len(pool)
        )
        # New minimal profiling log (A/B/C/D)
        denom = block_vec_hit + block_vec_miss
        block_hit_rate = (block_vec_hit / denom) if denom > 0 else 1.0
        mode_label = "Enhanced" if use_enhanced else "Baseline"
        mx.logger.info(
            "🔍 VectorCache stats user_id=%s qvec_hit=%s block_hit=%d block_miss=%d block_hit_rate=%.4f path=%s file_mb=%.2f keys_before=%d keys_after=%d",
            str(user_id),
            str(qvec_cache_hit),
            block_vec_hit,
            block_vec_miss,
            block_hit_rate,
            store.path,
            vector_file_size_mb,
            store_key_count_before,
            store_key_count_after,
        )
        mx.logger.info(
            "PROFILE_D mode=%s user_id=%s q_id=%s pool_after=%d D_loop=%.6fs D_sort=%.6fs",
            mode_label,
            str(user_id),
            str(q_id),
            pool_size_after,
            t_loop,
            t_sort,
        )
        # Get target boxes for evaluation
        target_boxes = mx.evidence_to_targets(qa.get("evidence"), pool)

        return rankings, sim_map, target_boxes, t_total

    def run(
        self,
        result_jsonl: str,
        result_csv: str,
        use_enhanced: bool = True
    ):
        """
        Run retrieval on all queries.

        Args:
            result_jsonl: Output JSONL file path
            result_csv: Output CSV file path
            use_enhanced: If True, use enhanced retrieval with metadata filtering
        """
        mx = _mx()

        if not os.path.exists(mx.Config.RAW_DATA_FILE):
            mx.logger.error("❌ No raw data file.")
            return

        self.load()

        mode_label = "Enhanced" if use_enhanced else "Baseline"
        mx.logger.info("ℹ️ %s Retrieval will append results to: %s, %s", mode_label, result_jsonl, result_csv)

        header_written = os.path.exists(result_csv)
        os.makedirs(os.path.dirname(result_csv), exist_ok=True)

        csv_file = open(result_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if not header_written:
            writer.writerow([
                "User_ID",
                "QA_ID",
                "Question",
                "Category",
                "Ranking_ContentEventTopicKW",
                "Targets",
                "Mode",
            ])

        with open(mx.Config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
            raw_list = json.load(f)[: mx.Config.LIMIT_CONVERSATIONS]

        for data in raw_list:
            user_id = data.get("user_id")
            # For LoCoMo data: assign default user_id based on filename if missing
            if user_id is None:
                # Extract filename from RAW_DATA_FILE (e.g., "locomo10" from "locomo10.json")
                filename = os.path.basename(mx.Config.RAW_DATA_FILE)
                user_id = os.path.splitext(filename)[0]  # Remove .json extension
                mx.logger.info(f"ℹ️ Assigned user_id='{user_id}' for LoCoMo data entry")

            if user_id is None:
                mx.logger.warning("⚠️ Skipping entry without user_id")
                continue

            for qa_idx, qa in enumerate(data.get("qa", [])):
                if qa.get("category") == 5:
                    continue

                rankings, sim_map, target_boxes, _ = self._score_and_rank(
                    user_id, qa, use_enhanced=use_enhanced, qa_idx=qa_idx
                )

                graph_info = None
                if self.graph_expand and rankings.get("content_event_topic_kw"):
                    ranked = rankings.get("content_event_topic_kw", [])
                    if self.top_k is not None and self.top_k > 0:
                        ranked = ranked[: self.top_k]
                    graph_info = self._expand_graph_for_blocks(user_id, ranked)

                writer.writerow([
                    user_id,
                    qa_idx,
                    qa.get("question", ""),
                    qa.get("category", ""),
                    rankings.get("content_event_topic_kw", []),
                    target_boxes,
                    mode_label,
                ])

                res_entry = {
                    "user_id": user_id,
                    "qa_idx": qa_idx,
                    "question": qa.get("question", ""),
                    "category": qa.get("category", ""),
                    "rankings": rankings,
                    "target_boxes": target_boxes,
                    "mode": mode_label,
                }
                if graph_info is not None:
                    res_entry["graph"] = graph_info
                mx.TraceLogger.log(result_jsonl, res_entry)

            mx.logger.info("✅ %s retrieval done for user %s", mode_label, user_id)

        csv_file.close()
        self._flush_all_stores()
        mx.logger.info("✅ %s Retrieval results appended to %s", mode_label, result_jsonl)
