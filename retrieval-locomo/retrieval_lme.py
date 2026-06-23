"""
LongMemEval-specific retrieval module.

Key differences from the LoCoMo version:
1. load() merges boxes from multiple boxes_dirs, deduplicating by user_id
   (last directory wins, so gpt2 overwrites gpt for the overlapping user 1192316e).
2. evidence_to_targets_lme() uses answer_session_ids directly against
   coverage.session_id (no "D1:3" format needed).
3. _score_and_rank() reads qa["_base_time"] (a datetime) and passes it to
   QueryParser.parse() so relative time expressions resolve correctly.
4. vector_store paths are per-user based on the source run's VECTOR_DIR;
   LMERetriever maintains a uid->vector_dir mapping and switches
   Config.VECTOR_DIR before constructing EmbeddingStore for each user.
5. run() iterates over the lme_data_dir, skipping questions with no blocks.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import time as _time_mod
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics.pairwise import cosine_similarity

# Resolve parent directory so retrieval/ can import from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_pasing_byllm import QueryParser, SearchDirective, dispatch_temporal_filter
from interval_tree_index import TemporalIndex
from anchor_resolver import AnchorResolver

try:
    from graph_storage import GraphConfig, MemgraphEventGraph
except Exception:
    GraphConfig = None  # type: ignore
    MemgraphEventGraph = None  # type: ignore


def _mx():
    """Lazy import to avoid circular imports."""
    import memblock_extractor as mx
    return mx


# ---------------------------------------------------------------------------
# LME-specific evidence helper
# ---------------------------------------------------------------------------

def evidence_to_targets_lme(
    answer_session_ids: List[str],
    pool: List[Dict[str, Any]],
) -> List[int]:
    """Map answer_session_ids to block_ids in pool.

    Unlike the LoCoMo version (which parses "D1:3" references), LME provides
    explicit session IDs.  We simply collect every block whose
    coverage.session_id matches one of the answer_session_ids.
    """
    if not answer_session_ids:
        return []
    answer_set = set(answer_session_ids)
    targets: List[int] = []
    seen: set = set()
    mx = _mx()
    for b in pool:
        sid = (b.get("coverage") or {}).get("session_id")
        if sid and sid in answer_set:
            bid = mx._get_block_id(b)
            if bid not in seen:
                targets.append(bid)
                seen.add(bid)
    return sorted(targets)


# ---------------------------------------------------------------------------
# LME question date parser
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")


def parse_question_date(text: Optional[str]) -> Optional[datetime]:
    """Parse LME question_date like '2023/05/30 (Tue) 22:40'."""
    if not text:
        return None
    m = _DATE_RE.search(text)
    if not m:
        return None
    try:
        return datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# LMERetriever
# ---------------------------------------------------------------------------

class LMERetriever:
    """Enhanced retriever adapted for LongMemEval data format.

    Parameters
    ----------
    worker:
        LLMWorker instance (from memblock_extractor).
    boxes_dirs:
        List of (run_id, output_dir) pairs, e.g.:
            [("longmemeval_s_gpt",  "/data/lyc/SA-Mem/out/longmemeval_s_gpt"),
             ("longmemeval_s_gpt2", "/data/lyc/SA-Mem/out/longmemeval_s_gpt2")]
        Dirs are processed in order; later dirs overwrite earlier ones for
        duplicate user_ids.
    lme_data_dir:
        Directory containing per-question JSON files
        (e.g. /data/wjl/SA-Mem/data/lme_preprocessed).
    top_k:
        Maximum number of blocks to return in rankings (None = no limit).
    """

    def __init__(
        self,
        worker: Any,
        boxes_dirs: List[Tuple[str, str]],
        lme_data_dir: str,
        top_k: Optional[int] = None,
        graph_expand: bool = False,
        graph_min_score: float = 0.7,
        graph_limit: int = 200,
        graph_include_relations: bool = True,
        use_anchor: bool = True,
        axis_mode: str = "auto",
    ):
        mx = _mx()
        self.worker = worker
        self.boxes_dirs = boxes_dirs          # [(run_id, out_dir), ...]
        self.lme_data_dir = lme_data_dir
        self.top_k = mx.Config.TOP_K_RETRIEVE if top_k is None else top_k

        # Bi-temporal ablation switch (mirrors LoCoMo retriever).
        # - "auto"    : use _infer_time_axis from QueryParser (default).
        # - "session" : force axis=SESSION; ANCHOR queries degrade to full pool
        #               (skip per-user anchor_resolver construction in run()).
        # - "event"   : force axis=EVENT.
        # - "none"    : skip the entire temporal filtering block.
        axis_mode = (axis_mode or "auto").lower()
        if axis_mode not in {"auto", "session", "event", "none"}:
            raise ValueError(f"Invalid axis_mode={axis_mode!r}; must be auto/session/event/none.")
        self.axis_mode = axis_mode
        self.use_anchor = bool(use_anchor)

        self.query_parser = QueryParser(
            api_key=mx.Config.API_KEY,
            base_url=mx.Config.BASE_URL,
            model=mx.Config.LLM_MODEL,
        )

        # Populated by load()
        self.all_blocks: List[Dict[str, Any]] = []
        self.blocks_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.block_by_id: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.temporal_index: TemporalIndex = TemporalIndex()
        self.anchor_resolver: Optional[AnchorResolver] = None
        # uid -> vector_store directory (for cache path resolution)
        self.uid_to_vector_dir: Dict[str, str] = {}

        # Graph expansion (optional)
        self.graph_expand = bool(graph_expand)
        self.graph_min_score = float(graph_min_score)
        self.graph_limit = int(graph_limit)
        self.graph_include_relations = bool(graph_include_relations)
        self.graph: Optional[Any] = None

        # Per-query timing records (for paper experiments)
        self.timing_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Load & index
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Merge boxes from all boxes_dirs, dedup by user_id, build index."""
        mx = _mx()

        # Collect blocks per user_id; later dirs overwrite earlier ones.
        uid_to_run: Dict[str, str] = {}
        uid_to_blocks: Dict[str, List[Dict[str, Any]]] = {}
        uid_to_vdir: Dict[str, str] = {}

        for run_id, out_dir in self.boxes_dirs:
            jsonl_path = os.path.join(out_dir, "final_boxes_content.jsonl")
            if not os.path.exists(jsonl_path):
                mx.logger.warning("⚠️ boxes file not found: %s", jsonl_path)
                continue

            vdir = os.path.join(out_dir, "vector_store")
            run_blocks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    b = json.loads(line)
                    uid = b.get("user_id")
                    if uid is None:
                        continue
                    uid = str(uid)
                    # Skip blocks whose user_id is not a valid 8-char hex
                    # (guards against the locomo10 bug in gpt3).
                    if not re.match(r"^[0-9a-f]{8}$", uid):
                        mx.logger.debug("Skipping block with non-LME user_id=%s", uid)
                        continue
                    run_blocks[uid].append(b)

            for uid, blocks in run_blocks.items():
                prev = uid_to_run.get(uid)
                if prev:
                    mx.logger.info(
                        "ℹ️ Overwriting user_id=%s from run '%s' with run '%s' (%d blocks -> %d blocks)",
                        uid, prev, run_id,
                        len(uid_to_blocks.get(uid, [])), len(blocks),
                    )
                uid_to_blocks[uid] = blocks
                uid_to_run[uid] = run_id
                uid_to_vdir[uid] = vdir
            mx.logger.info("✅ Loaded run '%s': %d users from %s", run_id, len(run_blocks), jsonl_path)

        # Flatten into self.all_blocks and build group/index structures
        self.uid_to_vector_dir = uid_to_vdir
        for uid, blocks in uid_to_blocks.items():
            self.blocks_by_user[uid] = blocks
            self.all_blocks.extend(blocks)
            for b in blocks:
                bid = mx._get_block_id(b)
                if bid is not None:
                    self.block_by_id[(uid, int(bid))] = b

        # NOTE: Temporal index is built per-user in _score_and_rank()
        # to avoid Python RecursionError when inserting all 3000+ blocks
        # into a single IntervalTree.

        mx.logger.info(
            "✅ Merged blocks: %d users, %d total blocks",
            len(uid_to_blocks), len(self.all_blocks),
        )

        # Optional graph init
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

    # ------------------------------------------------------------------
    # Temporal filtering (same logic as retrieval_enhanced_locomo.py)
    # ------------------------------------------------------------------

    def _filter_by_metadata(
        self,
        pool: List[Dict[str, Any]],
        directive: SearchDirective,
        user_ti: "TemporalIndex",
        user_anchor_resolver: "AnchorResolver",
    ) -> List[Dict[str, Any]]:
        mx = _mx()
        t_temporal = 0.0
        t_anchor = 0.0

        # Map enumerate-index -> block (aligns with TemporalIndex.global_id)
        pool_idx_to_block: Dict[int, Dict[str, Any]] = {i: b for i, b in enumerate(pool)}
        pool_ids = set(pool_idx_to_block.keys())

        mx.logger.info("🔍 pool_ids size = %d blocks", len(pool_ids))
        mx.logger.info(
            "🔍 use_interval_tree=%s, time_constraint.type=%s",
            directive.use_interval_tree, directive.time_constraint.type,
        )

        def _temporal_query_ids(start_date, end_date, time_type, use_event_time):
            ids = set(
                user_ti.query_temporal(
                    start_date=start_date,
                    end_date=end_date,
                    time_type=time_type,
                    use_event_time=use_event_time,
                )
            )
            return ids & pool_ids

        if directive.intent in {"PLANNING", "STATIC"}:
            mx.logger.info(
                "🔮 %s query: skipping temporal filtering.", directive.intent
            )

        t_temporal_start = _time_mod.perf_counter()
        time_filtered_ids = pool_ids

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
                elif not self.use_anchor or user_anchor_resolver is None:
                    mx.logger.info("🔍 ANCHOR query but anchor resolution disabled → full pool")
                    time_filtered_ids = pool_ids
                else:
                    mx.logger.info("🔍 ANCHOR: '%s' %s", tc.anchor_event, tc.anchor_relation)
                    t_a0 = _time_mod.perf_counter()
                    anchor_start, anchor_end, anchor_type = user_anchor_resolver.resolve_anchor(
                        tc.anchor_event or tc.raw_text or "",
                        tc.anchor_relation or "DURING",
                        pool[0].get("user_id") if pool else None,
                        pool,
                    )
                    t_anchor = _time_mod.perf_counter() - t_a0
                    if anchor_type != "NONE" and anchor_start:
                        ids_evt = _temporal_query_ids(
                            anchor_start.isoformat(),
                            anchor_end.isoformat() if anchor_end else None,
                            anchor_type,
                            True,
                        )
                        mx.logger.info(
                            "🔍 Anchor(event) type=%s [%s, %s] -> %d/%d blocks",
                            anchor_type, anchor_start, anchor_end,
                            len(ids_evt), len(pool_ids),
                        )
                        # Merge the anchor blocks themselves into the filtered
                        # pool so that temporal-reasoning queries retain BOTH
                        # the anchor event side and the time-window side.
                        anchor_pool_ids: set = set()
                        for ab in getattr(user_anchor_resolver, "last_anchor_blocks", []):
                            for pi, pb in enumerate(pool):
                                if pb is ab:
                                    anchor_pool_ids.add(pi)
                                    break
                        if anchor_pool_ids:
                            mx.logger.info(
                                "🔍 Merging %d anchor block(s) into filtered pool",
                                len(anchor_pool_ids),
                            )
                        time_filtered_ids = ids_evt | anchor_pool_ids
                    else:
                        mx.logger.warning("⚠️ Anchor resolution failed, using full pool")
                        time_filtered_ids = pool_ids
            else:
                if self.axis_mode == "session":
                    axis = "SESSION"
                elif self.axis_mode == "event":
                    axis = "EVENT"
                # axis_mode == "auto" keeps QueryParser-inferred axis.
                time_filtered_ids, mode_used = dispatch_temporal_filter(
                    axis,
                    query_event=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, True),
                    query_session=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, False),
                )
                mx.logger.info(
                    "🔍 Temporal (axis=%s, mode=%s, axis_mode=%s): %s [%s, %s] -> %d/%d",
                    axis, mode_used, self.axis_mode, tc.type, tc.start, tc.end,
                    len(time_filtered_ids), len(pool_ids),
                )

                # P0 LME fix: small-pool fallback.
                # If the filtered pool is too small (likely the literal time
                # window missed the session-time spillover that LME has a lot
                # of), first try BOTH_UNION (axis=BOTH_UNION), and if that's
                # still tiny, degrade to the full pool. The threshold mimics
                # "we want at least ~K candidates for the vector ranker".
                # Only triggers in axis_mode=auto so per-axis ablation results
                # remain pure.
                small_thresh = int(os.environ.get("LME_SMALL_POOL_THRESH", "20"))
                small_ratio = float(os.environ.get("LME_SMALL_POOL_RATIO", "0.1"))
                if (
                    self.axis_mode == "auto"
                    and len(time_filtered_ids) < small_thresh
                    and len(time_filtered_ids) < small_ratio * max(1, len(pool_ids))
                    and axis != "BOTH_UNION"
                ):
                    union_ids, _ = dispatch_temporal_filter(
                        "BOTH_UNION",
                        query_event=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, True),
                        query_session=lambda: _temporal_query_ids(tc.start, tc.end, tc.type, False),
                    )
                    if len(union_ids) > len(time_filtered_ids):
                        mx.logger.info(
                            "🩹 [small-pool fallback] axis=%s -> BOTH_UNION: %d -> %d",
                            axis, len(time_filtered_ids), len(union_ids),
                        )
                        time_filtered_ids = union_ids
                if (
                    self.axis_mode == "auto"
                    and len(time_filtered_ids) < small_thresh
                    and len(time_filtered_ids) < small_ratio * max(1, len(pool_ids))
                ):
                    mx.logger.info(
                        "🩹 [small-pool fallback] still too small (%d/%d) -> full pool",
                        len(time_filtered_ids), len(pool_ids),
                    )
                    time_filtered_ids = pool_ids

        t_temporal = _time_mod.perf_counter() - t_temporal_start
        mx.logger.info("🔍 After temporal filter: %d blocks", len(time_filtered_ids))

        # Safety fallback
        if len(time_filtered_ids) == 0:
            mx.logger.warning(
                "⚠️ 0 blocks after filtering -> falling back to full pool (%d blocks)",
                len(pool_ids),
            )
            time_filtered_ids = pool_ids

        mx.logger.info(
            "⏱️ Filter timing: temporal=%.3fs (anchor=%.3fs)",
            t_temporal, t_anchor,
        )

        return [pool_idx_to_block[idx] for idx in time_filtered_ids if idx in pool_idx_to_block]

    # ------------------------------------------------------------------
    # Score and rank
    # ------------------------------------------------------------------

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
            for key in ("sessionend_time", "sessionstart_time"):
                dt = self._parse_iso_to_datetime(tdx.get(key))
                if dt is not None and (latest is None or dt > latest):
                    latest = dt
        return latest if latest is not None else datetime.now()

    def _score_and_rank(
        self,
        user_id: str,
        qa: Dict[str, Any],
        use_enhanced: bool = True,
    ) -> Tuple[Dict[str, List[int]], Dict[int, float], List[int], Dict[str, Any], Dict[str, Any]]:
        """Score and rank blocks for a single question.

        qa must have:
          - "question"           : str
          - "answer_session_ids" : List[str]   (LME-style evidence)
          - "_base_time"         : datetime or None  (from question_date)
          - "question_id"        : str  (used as q_id in cache keys)
        """
        mx = _mx()

        pool = self.blocks_by_user.get(user_id, [])
        empty_timing = {
            "parse_source": "NONE",
            "t_parse": 0.0, "t_filter": 0.0, "t_rank": 0.0,
            "search_latency": 0.0, "t_total": 0.0,
        }
        empty_meta = {
            "time_constraint_type": "NONE",
            "query_time_start": None, "query_time_end": None,
        }
        if not pool:
            mx.logger.warning("⚠️ No blocks for user_id=%s", user_id)
            return {}, {}, [], empty_meta, empty_timing

        mx.logger.info("🔍 Initial pool: %d blocks for user_id=%s", len(pool), user_id)

        question = qa.get("question", "") or ""
        q_id = qa.get("question_id", qa.get("id", question[:40]))
        base_time: Optional[datetime] = qa.get("_base_time")

        t_parse = t_filter = t_rank = 0.0

        # ---- Query parsing ----
        directive = None
        if use_enhanced and question:
            try:
                if base_time is None:
                    base_time = self._infer_base_time_from_pool(pool)
                t0 = _time_mod.perf_counter()
                directive = self.query_parser.parse(question, base_time=base_time)
                t_parse = _time_mod.perf_counter() - t0
                mx.logger.info(
                    "📝 Parsed: intent=%s, time=%s, source=%s",
                    directive.intent, directive.time_constraint.type, directive.parse_source,
                )
            except Exception as e:
                mx.logger.warning("⚠️ Query parsing failed: %s", e)
                directive = None

        # ---- Metadata filtering ----
        if directive and use_enhanced:
            t0 = _time_mod.perf_counter()
            # Build a per-user TemporalIndex (pool-only) to avoid the global
            # RecursionError that occurs when all 3000+ blocks are in one tree.
            user_ti = TemporalIndex()
            user_ti.build_from_blocks(pool)
            # Skip per-user anchor_resolver when --no-use-anchor or axis_mode=session.
            if self.use_anchor and self.axis_mode != "session":
                user_anchor_resolver = AnchorResolver(user_ti, self.worker)
            else:
                user_anchor_resolver = None
            filtered_pool = self._filter_by_metadata(pool, directive, user_ti, user_anchor_resolver)
            t_filter = _time_mod.perf_counter() - t0
            mx.logger.info(
                "🔍 Metadata filter: %d/%d blocks", len(filtered_pool), len(pool)
            )
        else:
            filtered_pool = pool

        # ---- Vector ranking ----
        # Switch Config.VECTOR_DIR so EmbeddingStore loads the right cache file.
        vdir = self.uid_to_vector_dir.get(user_id)
        if vdir:
            mx.Config.VECTOR_DIR = vdir
        t_rank_start = _time_mod.perf_counter()
        store = mx.EmbeddingStore(self.worker, user_id)

        query_text = directive.rewritten_query if directive else question
        qvec = store.get_vector(
            f"qa_{user_id}_{q_id}",
            "question",
            query_text,
            note=f"U{user_id}_QA_LME",
        )

        mx.logger.info("🔄 Computing similarity for %d blocks...", len(filtered_pool))
        sim_map: Dict[int, float] = {}
        for idx, b in enumerate(filtered_pool):
            bid = mx._get_block_id(b)
            key = f"{user_id}_{bid}"

            features = b.get("features", {})
            content_text = features.get("content_text", "")
            topic_kw = features.get("topic_kw_text", "")
            events = b.get("events", [])
            event_texts = [e.get("description", "") for e in events if e.get("description")]
            event_str = " | ".join(event_texts[:20])
            text = f"{content_text} {topic_kw} {event_str}".strip()

            v = store.get_vector(
                key,
                "content_event_topic_kw",
                text,
                note=f"U{user_id}_B{bid}_lme",
            )
            try:
                s = cosine_similarity([qvec], [v])[0][0] if v else -1.0
            except Exception:
                s = -1.0
            sim_map[bid] = float(s)

        mx.logger.info("✅ Similarity done for %d blocks", len(filtered_pool))
        ranked = [bid for bid, _ in sorted(sim_map.items(), key=lambda x: x[1], reverse=True)]
        rankings = {"content_event_topic_kw": ranked}
        store.flush()
        t_rank = _time_mod.perf_counter() - t_rank_start

        t_search = t_filter + t_rank
        t_total = t_parse + t_search
        mx.logger.info(
            "⏱️ parse=%.3fs filter=%.3fs rank=%.3fs search=%.3fs total=%.3fs",
            t_parse, t_filter, t_rank, t_search, t_total,
        )

        parse_source = directive.parse_source if directive else "NONE"
        timing_info = {
            "parse_source": parse_source,
            "t_parse": t_parse, "t_filter": t_filter, "t_rank": t_rank,
            "search_latency": t_search, "t_total": t_total,
        }
        self.timing_records.append({
            "user_id": user_id,
            "q_id": str(q_id),
            "question": question,
            **timing_info,
        })

        # ---- target boxes (LME-specific) ----
        answer_session_ids = qa.get("answer_session_ids", []) or []
        target_boxes = evidence_to_targets_lme(answer_session_ids, pool)

        if directive:
            tc = directive.time_constraint
            query_time_meta = {
                "time_constraint_type": tc.type,
                "query_time_start": tc.start,
                "query_time_end": tc.end,
            }
        else:
            query_time_meta = empty_meta.copy()

        return rankings, sim_map, target_boxes, query_time_meta, timing_info

    # ------------------------------------------------------------------
    # Block time window helper (for minimal result rows)
    # ------------------------------------------------------------------

    def _extract_block_time_window(
        self, user_id: str, block_id: int
    ) -> Tuple[Optional[str], Optional[str]]:
        block = self.block_by_id.get((user_id, int(block_id)))
        if not block:
            return None, None
        tdx = block.get("temporal_index", {}) or {}
        block_start = tdx.get("block_event_start_time") or tdx.get("sessionstart_time")
        block_end = tdx.get("block_event_end_time") or tdx.get("sessionend_time")
        return block_start, block_end

    def _build_retrieved_item_minimal_rows(
        self, user_id: str, ranked_ids: List[int]
    ) -> List[Dict[str, Any]]:
        rows = []
        for rank, bid in enumerate(ranked_ids, start=1):
            bstart, bend = self._extract_block_time_window(user_id, int(bid))
            rows.append({
                "block_id": int(bid),
                "rank": rank,
                "block_time_start": bstart,
                "block_time_end": bend,
            })
        return rows

    # ------------------------------------------------------------------
    # Graph expansion (unchanged from locomo version)
    # ------------------------------------------------------------------

    def _expand_graph_for_blocks(
        self, user_id: Any, block_ids: List[Any]
    ) -> Dict[str, Any]:
        mx = _mx()
        if not self.graph or not block_ids:
            return {}
        try:
            events_by_block = self.graph.get_events_by_block_ids(
                user_id=str(user_id), block_ids=block_ids,
            )
            seed_event_ids: List[str] = []
            seen_seed: set = set()
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
            neighbor_ids: set = set()
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
                    edge_map[key] = {"from_event_id": a, "to_event_id": b, "score": score}
                if a in seed_set and b not in seed_set:
                    neighbor_ids.add(b)
                if b in seed_set and a not in seed_set:
                    neighbor_ids.add(a)

            expanded_event_ids = [eid for eid in neighbor_ids if eid not in seed_set]
            expanded_events = self.graph.get_events_by_ids(
                user_id=str(user_id), event_ids=expanded_event_ids,
            )
            relations: List[Dict[str, Any]] = []
            if self.graph_include_relations:
                relation_event_ids = list(seed_set | set(expanded_event_ids))
                relations = self.graph.get_relations_by_event_ids(
                    user_id=str(user_id),
                    event_ids=relation_event_ids,
                    limit=self.graph_limit,
                )
            rel_seen: set = set()
            rel_out: List[Dict[str, Any]] = []
            for rel in relations or []:
                key2 = (
                    str(rel.get("source") or ""),
                    str(rel.get("relationship") or ""),
                    str(rel.get("destination") or ""),
                    str(rel.get("event_id") or ""),
                )
                if key2 in rel_seen:
                    continue
                rel_seen.add(key2)
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

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        result_jsonl: str,
        result_csv: str,
        use_enhanced: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        """Run retrieval over all LME questions that have blocks.

        Parameters
        ----------
        result_jsonl:
            Output JSONL path for detailed results.
        result_csv:
            Output CSV path for summary results.
        use_enhanced:
            If True, use temporal filtering + vector ranking.
            If False, pure vector ranking (baseline).
        limit:
            If set, process at most this many questions (for debugging).
        """
        mx = _mx()

        self.load()

        mode_label = "Enhanced" if use_enhanced else "Baseline"
        mx.logger.info(
            "ℹ️ %s Retrieval → %s, %s", mode_label, result_jsonl, result_csv
        )
        mx.logger.info(
            "ℹ️ users with blocks: %d", len(self.blocks_by_user)
        )

        os.makedirs(os.path.dirname(os.path.abspath(result_csv)), exist_ok=True)
        header_written = os.path.exists(result_csv)
        csv_file = open(result_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if not header_written:
            writer.writerow([
                "User_ID",
                "Question_ID",
                "Question_Type",
                "Question",
                "Answer",
                "Ranking_ContentEventTopicKW",
                "Target_Boxes",
                "Mode",
            ])

        # Collect all per-question JSON files from lme_data_dir.
        # Accept only 8-char hex filenames (no _abs variants).
        lme_files = sorted(
            p for p in glob.glob(os.path.join(self.lme_data_dir, "*.json"))
            if re.match(r"^[0-9a-f]{8}\.json$", os.path.basename(p))
        )
        mx.logger.info(
            "ℹ️ Found %d LME question files in %s", len(lme_files), self.lme_data_dir
        )

        processed = 0
        skipped = 0
        for lme_path in lme_files:
            if limit is not None and processed >= limit:
                break

            with open(lme_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not items:
                continue

            item = items[0]  # Each file has exactly one question item
            user_id = item.get("question_id", "")
            if not user_id:
                skipped += 1
                continue

            # Skip questions with no blocks
            if user_id not in self.blocks_by_user:
                mx.logger.debug("⏭ No blocks for question_id=%s, skipping", user_id)
                skipped += 1
                continue

            question = item.get("question", "") or ""
            question_type = item.get("question_type", "")
            answer = item.get("answer", "") or ""
            answer_session_ids = item.get("answer_session_ids", []) or []
            question_date = item.get("question_date")
            base_time = parse_question_date(question_date)

            if base_time is None:
                mx.logger.warning(
                    "⚠️ Could not parse question_date='%s' for %s; "
                    "falling back to pool inference.",
                    question_date, user_id,
                )

            qa = {
                "question_id": user_id,
                "question": question,
                "question_type": question_type,
                "answer": answer,
                "answer_session_ids": answer_session_ids,
                "_base_time": base_time,
            }

            mx.logger.info(
                "🔍 Processing question_id=%s type=%s question='%.60s...'",
                user_id, question_type, question,
            )

            rankings, _sim_map, target_boxes, query_time_meta, _timing = self._score_and_rank(
                user_id, qa, use_enhanced=use_enhanced
            )

            ranked_ids = rankings.get("content_event_topic_kw", [])
            retrieved_items_minimal = self._build_retrieved_item_minimal_rows(user_id, ranked_ids)

            graph_info = None
            if self.graph_expand and ranked_ids:
                top_ids = ranked_ids[: self.top_k] if self.top_k else ranked_ids
                graph_info = self._expand_graph_for_blocks(user_id, top_ids)

            writer.writerow([
                user_id,
                user_id,
                question_type,
                question,
                answer,
                ranked_ids,
                target_boxes,
                mode_label,
            ])

            res_entry = {
                "user_id": user_id,
                "question_id": user_id,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "answer_session_ids": answer_session_ids,
                "question_date": question_date,
                "rankings": rankings,
                "target_boxes": target_boxes,
                "mode": mode_label,
                "time_constraint_type": query_time_meta.get("time_constraint_type"),
                "query_time_start": query_time_meta.get("query_time_start"),
                "query_time_end": query_time_meta.get("query_time_end"),
                "retrieved_items_minimal": retrieved_items_minimal,
            }
            if graph_info is not None:
                res_entry["graph"] = graph_info

            mx.TraceLogger.log(result_jsonl, res_entry)
            processed += 1
            mx.logger.info(
                "✅ Done user_id=%s  target_boxes=%s  top5=%s",
                user_id, target_boxes, ranked_ids[:5],
            )

        csv_file.close()
        mx.logger.info(
            "✅ %s Retrieval complete: processed=%d skipped=%d → %s",
            mode_label, processed, skipped, result_jsonl,
        )

        # Save per-query timing records
        timing_jsonl = result_jsonl.replace(".jsonl", "_timings.jsonl")
        self._print_timing_summary(save_path=timing_jsonl)

    # ------------------------------------------------------------------
    # Timing summary (same as locomo version)
    # ------------------------------------------------------------------

    def _print_timing_summary(self, save_path: Optional[str] = None) -> None:
        import statistics
        mx = _mx()

        records = self.timing_records
        if not records:
            mx.logger.info("ℹ️ No timing records to summarise.")
            return

        def _summarise(vals):
            if not vals:
                return {}
            return {
                "n": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "p95": sorted(vals)[int(len(vals) * 0.95)],
                "max": max(vals),
            }

        by_source: Dict[str, List] = defaultdict(list)
        for r in records:
            by_source[r["parse_source"]].append(r["t_parse"])

        summary = {
            "total_queries": len(records),
            "parse_source_counts": {k: len(v) for k, v in by_source.items()},
            "t_parse_by_source": {k: _summarise(v) for k, v in by_source.items()},
            "t_filter": _summarise([r["t_filter"] for r in records]),
            "t_rank": _summarise([r["t_rank"] for r in records]),
            "search_latency": _summarise([r["search_latency"] for r in records]),
            "t_total": _summarise([r["t_total"] for r in records]),
        }
        mx.logger.info("⏱️ Timing summary: %s", json.dumps(summary, indent=2))

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")
            mx.logger.info("⏱️ Timing records saved to %s", save_path)
