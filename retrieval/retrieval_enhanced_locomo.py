"""
Enhanced retrieval with query parsing, time-based filtering, and topic-based filtering.
Integrates query_pasing_byllm.py for intent recognition and metadata extraction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from query_pasing_byllm import QueryParser, SearchDirective, dispatch_temporal_filter
from interval_tree_index import TemporalIndex
from anchor_resolver import AnchorResolver
from datetime import datetime

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
    3. Category/topic-based filtering
    4. Vector similarity ranking
    """

    def __init__(
        self,
        worker: Any,
        top_k: int = None,
        graph_expand: bool = False,
        graph_min_score: float = 0.7,
        graph_limit: int = 200,
        graph_hops: int = 1,
        graph_include_relations: bool = True,
        graph_person_relations: bool = False,
        axis_mode: str = "auto",
    ):
        mx = _mx()
        self.worker = worker
        self.top_k = mx.Config.TOP_K_RETRIEVE if top_k is None else top_k

        # Query parser for intent and time extraction
        self.query_parser = QueryParser(
            api_key=mx.Config.API_KEY,
            base_url=mx.Config.BASE_URL,
            model=mx.Config.LLM_MODEL
        )

        # Data structures
        self.all_blocks: List[Dict[str, Any]] = []
        self.block_by_id: Dict[Any, Dict[str, Any]] = {}
        self.temporal_index: TemporalIndex = TemporalIndex()
        self.blocks_by_user: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        self.trace_map: Dict[Any, Dict[int, List[int]]] = {}
        self.anchor_resolver: Optional[AnchorResolver] = None

        # Bi-temporal ablation switch
        axis_mode = (axis_mode or "auto").lower()
        if axis_mode not in {"auto", "session", "event", "none"}:
            raise ValueError(f"Invalid axis_mode={axis_mode!r}")
        self.axis_mode = axis_mode

        # Graph expansion config (optional)
        self.graph_expand = bool(graph_expand)
        self.graph_min_score = float(graph_min_score)
        self.graph_limit = int(graph_limit)
        self.graph_hops = max(1, int(graph_hops))
        self.graph_include_relations = bool(graph_include_relations)
        self.graph_person_relations = bool(graph_person_relations)
        self.graph_extract_source = str(mx.Config.GRAPH_EXTRACT_SOURCE or "event").strip().lower()
        self.graph: Optional[Any] = None

        # Per-query latency tracking
        self.timing_records: List[Dict[str, Any]] = []

    def load(self):
        """Load memory blocks and build indices."""
        mx = _mx()

        # Load blocks
        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            self.all_blocks = [json.loads(l) for l in f if l.strip()]

        # Group by user
        for block in self.all_blocks:
            raw_user_id = block.get("user_id")
            if raw_user_id is None:
                continue
            user_id = str(raw_user_id)
            self.blocks_by_user[user_id].append(block)

            bid = mx._get_block_id(block)
            if bid is not None:
                # Key by (user_id, block_id) to avoid cross-user collisions
                self.block_by_id[(user_id, int(bid))] = block

        # Build temporal index
        self.temporal_index.build_from_blocks(self.all_blocks)

        # Initialize anchor resolver
        self.anchor_resolver = AnchorResolver(self.temporal_index, self.worker)

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

    def _relation_fact_text(self, rel: Dict[str, Any]) -> str:
        src = str(rel.get("source") or "").strip()
        rel_type = str(rel.get("relationship") or "").strip()
        dst = str(rel.get("destination") or "").strip()
        if not (src and rel_type and dst):
            return ""
        return f"{src} {rel_type} {dst}".strip()

    def _score_relation_facts(
        self,
        user_id: Any,
        query_text: str,
        relations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mx = _mx()
        if not query_text or not relations:
            return []
        store = mx.EmbeddingStore(self.worker, user_id)
        q_digest = hashlib.md5(query_text.encode("utf-8")).hexdigest()[:12]
        qvec = store.get_vector(
            f"qa_{user_id}_graph_profile_{q_digest}",
            "question",
            query_text,
            note=f"U{user_id}_QA_GraphProfile",
        )

        scored: List[Dict[str, Any]] = []
        for rel in relations:
            fact_text = self._relation_fact_text(rel)
            if not fact_text:
                continue
            digest = hashlib.md5(fact_text.encode("utf-8")).hexdigest()[:12]
            key = f"graph_rel_{user_id}_{digest}"
            v = store.get_vector(
                key,
                "relation_fact",
                fact_text,
                note=f"U{user_id}_GraphRelFact",
            )
            try:
                score = cosine_similarity([qvec], [v])[0][0] if v else -1.0
            except Exception:
                score = -1.0
            item = dict(rel)
            if "relation_id" not in item:
                item["relation_id"] = None
            item["fact_text"] = fact_text
            item["query_similarity"] = float(score)
            scored.append(item)

        store.flush()
        return scored

    def _expand_graph_for_blocks(self, user_id: Any, block_ids: List[Any], query_text: str | None = None) -> Dict[str, Any]:
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

            seed_set = set(seed_event_ids)
            similar_edges = self.graph.get_similar_events_1hop(
                user_id=str(user_id),
                event_ids=seed_event_ids,
                min_score=self.graph_min_score,
                limit=self.graph_limit,
            )
            edge_map: Dict[tuple, Dict[str, Any]] = {}
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

            reachable_rows = self.graph.get_similar_events_hops_with_scores(
                user_id=str(user_id),
                event_ids=seed_event_ids,
                hops=self.graph_hops,
                min_score=self.graph_min_score,
                limit=self.graph_limit,
            )
            expanded_meta: Dict[str, Dict[str, Any]] = {}
            for row in reachable_rows:
                eid = str(row.get("event_id") or "").strip()
                if not eid or eid in seed_set:
                    continue
                score = float(row.get("similarity_score") or 0.0)
                hop = int(row.get("min_hop") or 0)
                prev = expanded_meta.get(eid)
                if prev is None or score > float(prev.get("graph_similarity_score") or -1.0):
                    expanded_meta[eid] = {
                        "graph_similarity_score": score,
                        "graph_min_hop": hop,
                    }

            expanded_event_ids = sorted(
                expanded_meta.keys(),
                key=lambda x: (
                    int(expanded_meta[x].get("graph_min_hop") or 0),
                    -float(expanded_meta[x].get("graph_similarity_score") or 0.0),
                    x,
                ),
            )
            expanded_events = self.graph.get_events_by_ids(
                user_id=str(user_id),
                event_ids=expanded_event_ids,
            )
            for ev in expanded_events:
                eid = str(ev.get("event_id") or "").strip()
                if not eid:
                    continue
                meta = expanded_meta.get(eid)
                if not meta:
                    continue
                ev.update(meta)

            relations: List[Dict[str, Any]] = []
            if self.graph_include_relations:
                if self.graph_extract_source == "raw":
                    relations = self.graph.get_relations_by_block_ids(
                        user_id=str(user_id),
                        block_ids=block_ids,
                        limit=self.graph_limit,
                    )
                    relations = [r for rels in (relations or {}).values() for r in (rels or [])]
                else:
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

            person_relation_facts: List[Dict[str, Any]] = []
            query_text_clean = str(query_text or "").strip()
            if query_text_clean and self.graph_include_relations and self.graph_person_relations:
                ent_names = set()
                for rel in relations or []:
                    src = str(rel.get("source") or "").strip()
                    dst = str(rel.get("destination") or "").strip()
                    if src:
                        ent_names.add(src)
                    if dst:
                        ent_names.add(dst)

                entity_types = self.graph.get_entity_types(
                    user_id=str(user_id),
                    entity_names=sorted(ent_names),
                )
                person_names = sorted(
                    [
                        name
                        for name, etype in (entity_types or {}).items()
                        if str(etype or "").strip().lower() == "person"
                    ]
                )

                if person_names:
                    person_relations = self.graph.get_relations_for_entities(
                        user_id=str(user_id),
                        entity_names=person_names,
                        limit=self.graph_limit,
                    )
                    rel_seen = set()
                    rel_deduped: List[Dict[str, Any]] = []
                    for rel in person_relations or []:
                        key = (
                            str(rel.get("source") or ""),
                            str(rel.get("relationship") or ""),
                            str(rel.get("destination") or ""),
                        )
                        if key in rel_seen:
                            continue
                        rel_seen.add(key)
                        rel_deduped.append(rel)

                    person_relation_facts = self._score_relation_facts(
                        user_id=str(user_id),
                        query_text=query_text_clean,
                        relations=rel_deduped,
                    )
                    person_relation_facts.sort(
                        key=lambda r: float(r.get("query_similarity") or -1.0),
                        reverse=True,
                    )

                    profile_map: Dict[str, List[Dict[str, Any]]] = {name: [] for name in person_names}
                    for rel in person_relation_facts:
                        src = str(rel.get("source") or "").strip()
                        dst = str(rel.get("destination") or "").strip()
                        if src in profile_map:
                            profile_map[src].append(rel)
                        if dst in profile_map and dst != src:
                            profile_map[dst].append(rel)

                    grouped: List[Dict[str, Any]] = []
                    for name in person_names:
                        facts = profile_map.get(name, [])
                        if not facts:
                            continue
                        grouped.append(
                            {
                                "person": name,
                                "relation_facts": facts,
                            }
                        )
                    person_relation_facts = grouped

            return {
                "graph_enabled": True,
                "seed_event_ids": seed_event_ids,
                "expanded_events": expanded_events,
                "similar_events": list(edge_map.values()),
                "relations": rel_out,
                "person_relation_facts": person_relation_facts,
            }
        except Exception as e:
            mx.logger.warning("⚠️ Graph expansion failed: %s", e)
            return {"graph_enabled": False, "error": str(e)}

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
                    uid = str(uid)
                    trace_map.setdefault(uid, {})
                    for bid in ids:
                        trace_map[uid][int(bid)] = [int(x) for x in ids]
        except Exception:
            return {}
        return trace_map
    
    @staticmethod
    def _parse_iso_to_datetime(s: Optional[str]) -> Optional[datetime]:
        if not s or s == "Unknown":
            return None
        try:
            if "T" in s:
                return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            return datetime.strptime(str(s), "%Y-%m-%d")
        except Exception:
            return None

    def _infer_base_time_from_pool(self, pool: List[Dict[str, Any]]) -> datetime:
        latest: Optional[datetime] = None
        for b in pool:
            tdx = b.get("temporal_index", {}) or {}
            candidates = [
                tdx.get("sessionend_time"),
                tdx.get("sessionstart_time"),
            ]
            for s in candidates:
                dt = self._parse_iso_to_datetime(s)
                if dt is not None and (latest is None or dt > latest):
                    latest = dt
        return latest if latest is not None else datetime.now()

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

        # Use global block index in all_blocks as unique identifier.
        # IMPORTANT: keep one identifier system end-to-end in this function.
        block_to_idx = {id(b): i for i, b in enumerate(self.all_blocks)}
        pool_idx_to_block: Dict[int, Dict[str, Any]] = {}
        for b in pool:
            idx = block_to_idx.get(id(b))
            if idx is not None:
                pool_idx_to_block[idx] = b
        pool_ids = set(pool_idx_to_block.keys())

        # DEBUG: Log pool_ids size
        mx.logger.info("🔍 DEBUG: pool_ids size = %d blocks", len(pool_ids))

        # -------------------------
        # 1) Temporal filtering
        # -------------------------
        t_temporal_start = time.perf_counter()
        time_filtered_ids = pool_ids

        # DEBUG: Log temporal filtering decision
        mx.logger.info(
            "🔍 DEBUG: use_interval_tree=%s, time_constraint.type=%s",
            directive.use_interval_tree,
            directive.time_constraint.type,
        )

        def _temporal_query_ids(
            start_date: Optional[str],
            end_date: Optional[str],
            time_type: str,
            use_event_time: bool,
        ) -> set:
            ids = set(
                self.temporal_index.query_temporal(
                    start_date=start_date,
                    end_date=end_date,
                    time_type=time_type,
                    use_event_time=use_event_time,
                )
            )
            return ids & pool_ids

        # Log intent-based filtering decision
        if directive.intent in {"PLANNING", "STATIC"}:
            mx.logger.info(
                "🔮 %s query detected. Skipping temporal filtering (use_interval_tree=%s).",
                directive.intent,
                directive.use_interval_tree,
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
                        "🧪 [axis_mode=session] ANCHOR query → degrade to full pool."
                    )
                    time_filtered_ids = pool_ids
                elif not self.anchor_resolver:
                    mx.logger.info("🔍 ANCHOR query detected but no anchor resolver → using full pool")
                    time_filtered_ids = pool_ids
                else:
                    mx.logger.info("🔍 ANCHOR query detected: '%s' %s", tc.anchor_event, tc.anchor_relation)

                    t_anchor_start = time.perf_counter()
                    anchor_start, anchor_end = self.anchor_resolver.resolve_anchor(
                        tc.anchor_event or tc.raw_text or "",
                        tc.anchor_relation or "DURING",
                        pool[0].get("user_id") if pool else None,
                        pool,
                    )
                    t_anchor = time.perf_counter() - t_anchor_start

                    if anchor_start and anchor_end:
                        ids_evt = _temporal_query_ids(
                            anchor_start.isoformat(),
                            anchor_end.isoformat(),
                            "RANGE",
                            True,
                        )
                        mx.logger.info(
                            "🔍 Anchor temporal(event) [%s, %s] -> %d/%d blocks",
                            anchor_start,
                            anchor_end,
                            len(ids_evt),
                            len(pool_ids),
                        )
                        time_filtered_ids = ids_evt
                    else:
                        mx.logger.warning("⚠️ Anchor resolution failed, using full pool")
                        time_filtered_ids = pool_ids

            else:
                # Non-anchor temporal: axis-driven dispatcher
                if self.axis_mode == "session":
                    axis = "SESSION"
                elif self.axis_mode == "event":
                    axis = "EVENT"
                # axis_mode == "auto" keeps the QueryParser-inferred axis

                time_filtered_ids, mode_used = dispatch_temporal_filter(
                    axis,
                    query_event=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, True),
                    query_session=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, False),
                )
                mx.logger.info(
                    "🔍 Temporal filter (axis=%s, mode=%s, axis_mode=%s): %s [%s, %s] -> %d/%d blocks",
                    axis, mode_used, self.axis_mode, tc.type, tc.start, tc.end,
                    len(time_filtered_ids), len(pool_ids),
                )

        # DEBUG: Log time filtering result
        mx.logger.info("🔍 DEBUG: After temporal filtering, time_filtered_ids size = %d blocks", len(time_filtered_ids))
        t_temporal = time.perf_counter() - t_temporal_start

        # -------------------------
        # 2) Topic filtering - DISABLED
        # -------------------------
        category_filtered_ids = time_filtered_ids
        mx.logger.info("🔍 Topic filter: DISABLED (skipped)")

        # -------------------------
        # 3) Event type filtering - DISABLED
        # -------------------------
        t_event_type_start = time.perf_counter()
        type_filtered_ids = category_filtered_ids
        mx.logger.info("🔍 Event type filter: DISABLED (skipped)")
        t_event_type = time.perf_counter() - t_event_type_start

        # Final filtering result
        final_count = len(type_filtered_ids)
        if final_count == 0:
            mx.logger.warning("⚠️ Filtering resulted in 0 blocks. This query may not have matching information in memory.")
        elif final_count < 5:
            mx.logger.info("✅ Filtering resulted in %d blocks (small but precise result)", final_count)
        else:
            mx.logger.info("✅ Filtering resulted in %d blocks", final_count)

        # Log timing breakdown for metadata filtering
        mx.logger.info(
            "⏱️ Filter timing: temporal=%.3fs (anchor=%.3fs), event_type=%.3fs, total=%.3fs",
            t_temporal,
            t_anchor,
            t_event_type,
            t_temporal + t_event_type,
        )

        # Return using the SAME id system (global index in all_blocks).
        return [pool_idx_to_block[idx] for idx in type_filtered_ids if idx in pool_idx_to_block]

    def _score_and_rank(
        self,
        user_id: Any,
        qa: Dict[str, Any],
        use_enhanced: bool = True,
        store: Any = None,
    ) -> Tuple[Dict[str, List[int]], Dict[int, float], List[int], Dict[str, Any]]:
        """
        Score and rank blocks for a query with optional metadata filtering.

        Args:
            user_id: User identifier
            qa: Question-answer pair
            use_enhanced: If True, use query parsing and metadata filtering
            store: Optional pre-initialized EmbeddingStore (for warm-cache)

        Returns:
            Tuple of (rankings, similarity_map, target_boxes, query_time_meta)
        """
        mx = _mx()

        pool = self.blocks_by_user.get(user_id, [])
        if not pool:
            return {}, {}, [], {
                "time_constraint_type": "NONE",
                "query_time_start": None,
                "query_time_end": None,
            }

        mx.logger.info("🔍 DEBUG: Initial pool size = %d blocks", len(pool))

        question = qa.get("question", "") or ""
        q_id = qa.get("id", qa.get("question", ""))

        # Fine-grained timing variables
        import time
        t_parse = 0.0
        t_filter = 0.0
        t_store_init = 0.0
        t_query_vec = 0.0
        t_block_vec_fetch = 0.0
        t_cosine = 0.0
        t_sort = 0.0
        t_flush = 0.0
        fallback_to_full_pool = False
        initial_pool_size = len(pool)

        # Parse query if enhanced mode is enabled
        directive = None
        if use_enhanced and question:
            try:
                t_parse_start = time.perf_counter()
                base_time = self._infer_base_time_from_pool(pool)
                directive = self.query_parser.parse(question, base_time=base_time)
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
            if not filtered_pool:
                mx.logger.warning(
                    "Temporal filtering returned 0 blocks; falling back to full pool for semantic ranking."
                )
                filtered_pool = pool
                fallback_to_full_pool = True
        else:
            filtered_pool = pool

        # Vector similarity ranking on filtered pool
        t_rank_start = time.perf_counter()

        # Use provided store or create new one
        if store is None:
            t_store_init_start = time.perf_counter()
            store = mx.EmbeddingStore(self.worker, user_id)
            t_store_init = time.perf_counter() - t_store_init_start

        # Use rewritten query if available, otherwise use original
        query_text = directive.rewritten_query if directive else question
        t_query_vec_start = time.perf_counter()
        qvec = store.get_vector(
            f"qa_{user_id}_{q_id}",
            "question",
            query_text,
            note=f"U{user_id}_QA_Enhanced"
        )
        t_query_vec = time.perf_counter() - t_query_vec_start

        mx.logger.info("🔄 Computing similarity for %d blocks...", len(filtered_pool))
        sim_map: Dict[int, float] = {}
        for idx, b in enumerate(filtered_pool):
            if idx > 0 and idx % 50 == 0:
                mx.logger.info("   Progress: %d/%d blocks processed", idx, len(filtered_pool))

            bid = mx._get_block_id(b)
            key = f"{user_id}_{bid}"

            # Build enriched text for embedding (Membox style: content + events + topics)
            features = b.get("features", {})

            # Get content_text (original conversation)
            content_text = features.get("content_text", "")

            # Get topic keywords
            topic_kw = features.get("topic_kw_text", "")

            # Include event descriptions
            events = b.get("events", [])
            event_texts = [e.get("description", "") for e in events if e.get("description")]
            event_str = " | ".join(event_texts[:20])

            # Combine all: content_text + topic_kw + events (like Membox)
            text = f"{content_text} {topic_kw} {event_str}".strip()

            t_bv_start = time.perf_counter()
            v = store.get_vector(
                key,
                "content_event_topic_kw",
                text,
                note=f"U{user_id}_B{bid}_enhanced",
            )
            t_block_vec_fetch += time.perf_counter() - t_bv_start

            t_cosine_start = time.perf_counter()
            try:
                s = cosine_similarity([qvec], [v])[0][0] if v else -1.0
            except Exception:
                s = -1.0
            t_cosine += time.perf_counter() - t_cosine_start

            sim_map[bid] = float(s)

        mx.logger.info("✅ Similarity computation complete for %d blocks", len(filtered_pool))

        # Rank by similarity
        t_sort_start = time.perf_counter()
        ranked = [bid for bid, _ in sorted(sim_map.items(), key=lambda x: x[1], reverse=True)]
        t_sort = time.perf_counter() - t_sort_start

        rankings = {"content_event_topic_kw": ranked}

        t_flush_start = time.perf_counter()
        store.flush()
        t_flush = time.perf_counter() - t_flush_start

        t_rank = time.perf_counter() - t_rank_start

        # Log timing breakdown
        t_total = t_parse + t_filter + t_rank
        mx.logger.info(
            "⏱️ Timing: parse=%.3fs, filter=%.3fs, rank=%.3fs, total=%.3fs",
            t_parse, t_filter, t_rank, t_total
        )

        # Get target boxes for evaluation
        target_boxes = mx.evidence_to_targets(qa.get("evidence"), pool)

        query_time_meta = {
            "time_constraint_type": directive.time_constraint.type if directive else "NONE",
            "query_time_start": directive.time_constraint.start if directive else None,
            "query_time_end": directive.time_constraint.end if directive else None,
            "retrieval_latency_parse": t_parse,
            "retrieval_latency_filter": t_filter,
            "retrieval_latency_store_init": t_store_init,
            "retrieval_latency_query_vector": t_query_vec,
            "retrieval_latency_block_vector_fetch": t_block_vec_fetch,
            "retrieval_latency_cosine": t_cosine,
            "retrieval_latency_sort": t_sort,
            "retrieval_latency_flush": t_flush,
            "retrieval_latency_rank_total": t_rank,
            "retrieval_latency_search_no_parse": t_filter + t_rank,
            "retrieval_latency_total_with_parse": t_parse + t_filter + t_rank,
            "fallback_to_full_pool": fallback_to_full_pool,
            "filtered_pool_size": len(filtered_pool),
            "initial_pool_size": initial_pool_size,
            "parse_source": directive.parse_source if directive else "NONE",
            "query_intent": directive.intent if directive else "NONE",
            "time_axis": directive.time_axis if directive else "NONE",
        }

        return rankings, sim_map, target_boxes, query_time_meta

    def _extract_block_time_window(self, user_id: str, block_id: int) -> Tuple[Optional[str], Optional[str]]:
        block = self.block_by_id.get((user_id, int(block_id)))
        if not block:
            return None, None

        tdx = block.get("temporal_index", {}) or {}

        block_start = tdx.get("block_event_start_time") or tdx.get("sessionstart_time")
        block_end = tdx.get("block_event_end_time") or tdx.get("sessionend_time")

        return block_start, block_end

    def _build_retrieved_item_minimal_rows(
        self,
        user_id: str,
        ranked_ids: List[int],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for rank, bid in enumerate(ranked_ids, start=1):
            block_start, block_end = self._extract_block_time_window(user_id, int(bid))
            rows.append(
                {
                    "block_id": int(bid),
                    "rank": rank,
                    "block_time_start": block_start,
                    "block_time_end": block_end,
                }
            )
        return rows

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
            all_data = json.load(f)
            limit = mx.Config.LIMIT_CONVERSATIONS
            raw_list = all_data if (limit is None or limit <= 0) else all_data[:limit]
        # 读取文件时判断是 JSON 数组还是 JSONL，每行解析一个 JSON 对象
        # with open(mx.Config.RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        #     first = f.read(1)
        #     f.seek(0)
        #     if first == '[':
        #         raw_list = json.load(f)
        #     else:
        #         raw_list = [json.loads(line) for line in f if line.strip()]
        # raw_list = raw_list[: mx.Config.LIMIT_CONVERSATIONS]


        for conv_idx, data in enumerate(raw_list):
            user_id = str(conv_idx)  # Convert to string to match reindexed file format
            mx.logger.info(f"ℹ️ Assigned user_id={user_id} for conversation {conv_idx}")

            # Create EmbeddingStore once per user for warm-cache
            store = mx.EmbeddingStore(self.worker, user_id)

            qa_count_in_conv = 0

            for qa_idx, qa in enumerate(data.get("qa", [])):
                if qa.get("category") == 5:
                    continue

                qa_count_in_conv += 1

                rankings, _, target_boxes, query_time_meta = self._score_and_rank(
                    user_id, qa, use_enhanced=use_enhanced, store=store
                )

                ranked_ids = rankings.get("content_event_topic_kw", [])
                retrieved_items_minimal = self._build_retrieved_item_minimal_rows(user_id, ranked_ids)

                graph_info = None
                if self.graph_expand and rankings.get("content_event_topic_kw"):
                    ranked = rankings.get("content_event_topic_kw", [])
                    if self.top_k is not None and self.top_k > 0:
                        ranked = ranked[: self.top_k]
                    graph_info = self._expand_graph_for_blocks(
                        user_id,
                        ranked,
                        query_text=str(qa.get("question", "") or "").strip(),
                    )

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
                    "time_constraint_type": query_time_meta.get("time_constraint_type"),
                    "query_time_start": query_time_meta.get("query_time_start"),
                    "query_time_end": query_time_meta.get("query_time_end"),
                    "retrieved_items_minimal": retrieved_items_minimal,
                    "retrieval_latency_parse": query_time_meta.get("retrieval_latency_parse", 0.0),
                    "retrieval_latency_filter": query_time_meta.get("retrieval_latency_filter", 0.0),
                    "retrieval_latency_store_init": query_time_meta.get("retrieval_latency_store_init", 0.0),
                    "retrieval_latency_query_vector": query_time_meta.get("retrieval_latency_query_vector", 0.0),
                    "retrieval_latency_block_vector_fetch": query_time_meta.get("retrieval_latency_block_vector_fetch", 0.0),
                    "retrieval_latency_cosine": query_time_meta.get("retrieval_latency_cosine", 0.0),
                    "retrieval_latency_sort": query_time_meta.get("retrieval_latency_sort", 0.0),
                    "retrieval_latency_flush": query_time_meta.get("retrieval_latency_flush", 0.0),
                    "retrieval_latency_rank_total": query_time_meta.get("retrieval_latency_rank_total", 0.0),
                    "retrieval_latency_search_no_parse": query_time_meta.get("retrieval_latency_search_no_parse", 0.0),
                    "retrieval_latency_total_with_parse": query_time_meta.get("retrieval_latency_total_with_parse", 0.0),
                    "fallback_to_full_pool": query_time_meta.get("fallback_to_full_pool", False),
                    "filtered_pool_size": query_time_meta.get("filtered_pool_size", 0),
                    "initial_pool_size": query_time_meta.get("initial_pool_size", 0),
                    "parse_source": query_time_meta.get("parse_source", "NONE"),
                    "query_intent": query_time_meta.get("query_intent", "NONE"),
                    "time_axis": query_time_meta.get("time_axis", "NONE"),
                }
                if graph_info is not None:
                    res_entry["graph"] = graph_info
                mx.TraceLogger.log(result_jsonl, res_entry)

            # Flush store once per user (not per QA)
            store.flush()

            mx.logger.info(
                "✅ %s retrieval done for user %s (conversation %d: qa_idx %d-%d, %d queries)",
                mode_label, user_id, conv_idx, 0, qa_count_in_conv - 1, qa_count_in_conv
            )

        csv_file.close()
        mx.logger.info("✅ %s Retrieval results appended to %s", mode_label, result_jsonl)
