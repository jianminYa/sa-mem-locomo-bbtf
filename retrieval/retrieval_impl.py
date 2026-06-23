from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

from sklearn.metrics.pairwise import cosine_similarity
import time

try:
    from graph_storage import GraphConfig, MemgraphEventGraph
except Exception:  # pragma: no cover - optional dependency
    GraphConfig = None  # type: ignore
    MemgraphEventGraph = None  # type: ignore

def _mx():
    """
    Lazy import to avoid circular imports:
    memblock_extractor imports retrieval_impl, but retrieval_impl must not import memblock_extractor at import-time.
    """
    import memblock_extractor as mx  # local import by design

    return mx


class SimpleRetriever:
    """基于 content 向量与问题相似度的简易检索，不生成答案。"""

    def __init__(
        self,
        worker: Any,
        top_k: int = None,
        graph_expand: bool = False,
        graph_min_score: float = 0.7,
        graph_limit: int = 200,
        graph_hops: int | None = None,
        graph_include_relations: bool = True,
        graph_person_relations: bool | None = None,
    ):
        mx = _mx()
        self.worker = worker
        # top_k=None means keep full ranking; otherwise cap at provided value
        self.top_k = mx.Config.TOP_K_RETRIEVE if top_k is None else top_k
        self.all_boxes: List[Dict[str, Any]] = []
        self.trace_map: Dict[Any, Dict[int, List[int]]] = {}

        # Graph expansion config (optional)
        self.graph_expand = bool(graph_expand)
        self.graph_min_score = float(graph_min_score)
        self.graph_limit = int(graph_limit)
        _ = graph_hops
        self.graph_include_relations = bool(graph_include_relations)
        _ = graph_person_relations
        self.graph_extract_source = str(mx.Config.GRAPH_EXTRACT_SOURCE or "event").strip().lower()
        self.graph: Optional[Any] = None

    def load(self):
        mx = _mx()
        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            self.all_boxes = [json.loads(l) for l in f if l.strip()]
        self.trace_map = self._load_traces(mx.Config.TIME_TRACE_FILE)

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

    @staticmethod
    def _tokens(text: str) -> set:
        stop = {
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "for",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "with",
            "by",
            "at",
            "from",
            "that",
            "this",
            "it",
            "as",
            "but",
            "if",
            "about",
            "into",
            "than",
            "then",
            "so",
            "such",
            "not",
            "no",
            "do",
            "does",
            "did",
        }
        tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        return {t for t in tokens if t and t not in stop}

    def _load_traces(self, path: str) -> Dict[Any, Dict[int, List[int]]]:
        """Return mapping: user_id -> box_id -> trace box_ids from file."""
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

    def _score_and_rank(self, user_id: Any, qa: Dict[str, Any], qa_idx: int | None = None):
        mx = _mx()
        t_total_start = time.perf_counter()
        t_parse = 0.0   
        t_filter = 0.0 
        pool = [b for b in self.all_boxes if b.get("user_id") == user_id]
        q = qa.get("question", "") or ""
        q_id = qa.get("id", qa.get("question", ""))

        t_rank_start = time.perf_counter()
        store = mx.EmbeddingStore(self.worker, user_id)
        qvec = store.get_vector(f"qa_{user_id}_{q_id}", "question", q, note=f"U{user_id}_QA_Content")

        sim_map: Dict[int, float] = {}
        t_loop_start = time.perf_counter()
        for b in pool:
            bid = mx._get_block_id(b)
            key = f"{user_id}_{bid}"
            text = b.get("features", {}).get("content_text", "") or ""
            evt = mx._events_to_text(b.get("features", {}).get("events", []))
            text = f"{text} {evt} {b.get('features', {}).get('topic_kw_text', '')}".strip()

            v = store.get_vector(
                key,
                "content_event_topic_kw",
                text,
                note=f"U{user_id}_B{bid}_content_event_topic_kw",
            )
            try:
                s = cosine_similarity([qvec], [v])[0][0] if v else -1.0
            except Exception:
                s = -1.0
            sim_map[bid] = float(s)
        t_loop = time.perf_counter() - t_loop_start

        t_sort_start = time.perf_counter()
        ranked = [bid for bid, _ in sorted(sim_map.items(), key=lambda x: x[1], reverse=True)]
        t_sort = time.perf_counter() - t_sort_start
        # Keep full ordering for downstream reuse; top-k slice is optional
        rankings = {"content_event_topic_kw": ranked}

        store.flush()
        t_rank = time.perf_counter() - t_rank_start
        t_total = time.perf_counter() - t_total_start

        mx.logger.info(
            "PROFILE_D mode=%s user_id=%s q_id=%s pool_after=%d D_loop=%.6fs D_sort=%.6fs",
            "Baseline",
            str(user_id),
            str(q_id),
            len(pool),
            t_loop,
            t_sort,
        )
        target_boxes = mx.evidence_to_targets(qa.get("evidence"), pool)
        return rankings, sim_map, target_boxes

    def _expand_graph_for_blocks(self, user_id: Any, block_ids: List[Any]) -> Dict[str, Any]:
        mx = _mx()
        if not self.graph or not block_ids:
            return {}

        try:
            events_by_block = self.graph.get_events_by_block_ids(
                user_id=str(user_id),
                block_ids=block_ids,
            )
            all_event_ids = []
            for _, items in (events_by_block or {}).items():
                for ev in items:
                    eid = ev.get("event_id")
                    if eid:
                        all_event_ids.append(str(eid))

            similar_edges = self.graph.get_similar_events_1hop(
                user_id=str(user_id),
                event_ids=all_event_ids,
                min_score=self.graph_min_score,
                limit=self.graph_limit,
            )

            relations_by_block: Dict[str, List[Dict[str, Any]]] = {}
            if self.graph_include_relations:
                if self.graph_extract_source == "raw":
                    relations_by_block = self.graph.get_relations_by_block_ids(
                        user_id=str(user_id),
                        block_ids=block_ids,
                        limit=self.graph_limit,
                    )
                else:
                    relations = self.graph.get_relations_by_event_ids(
                        user_id=str(user_id),
                        event_ids=all_event_ids,
                        limit=self.graph_limit,
                    )
                    event_to_block: Dict[str, str] = {}
                    for bid_str, evs in (events_by_block or {}).items():
                        for ev in evs or []:
                            eid = str(ev.get("event_id") or "").strip()
                            if eid:
                                event_to_block[eid] = str(bid_str)
                    for rel in relations or []:
                        eid = str(rel.get("event_id") or "").strip()
                        bid_str = event_to_block.get(eid)
                        if not bid_str:
                            continue
                        relations_by_block.setdefault(bid_str, []).append(rel)

            events_to_edges: Dict[str, List[Dict[str, Any]]] = {}
            for edge in similar_edges or []:
                fe = str(edge.get("from_event_id") or "")
                if fe:
                    events_to_edges.setdefault(fe, []).append(edge)

            block_payload: Dict[str, Any] = {}
            for bid in block_ids:
                bid_str = str(bid)
                evs = events_by_block.get(bid_str, []) if events_by_block else []
                ev_ids = [str(e.get("event_id")) for e in evs if e.get("event_id")]
                edges = []
                for eid in ev_ids:
                    edges.extend(events_to_edges.get(eid, []))
                block_payload[bid_str] = {
                    "events": evs,
                    "similar_events": edges,
                    "relations": relations_by_block.get(bid_str, []) if relations_by_block else [],
                }

            return {
                "graph_enabled": True,
                "block_graph": block_payload,
            }
        except Exception as e:
            mx.logger.warning("⚠️ Graph expansion failed: %s", e)
            return {"graph_enabled": False, "error": str(e)}

    def run(self, result_jsonl: str, result_csv: str):
        mx = _mx()
        if not os.path.exists(mx.Config.RAW_DATA_FILE):
            mx.logger.error("❌ No raw data file.")
            return
        self.load()

        mx.logger.info("ℹ️ Retrieval will append results to: %s, %s", result_jsonl, result_csv)
        header_written = os.path.exists(result_csv)
        os.makedirs(os.path.dirname(result_csv), exist_ok=True)

        csv_file = open(result_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if not header_written:
            writer.writerow(
                [
                    "User_ID",
                    "QA_ID",
                    "Question",
                    "Category",
                    "Ranking_ContentEventTopicKW",
                    "Targets",
                ]
            )

        with open(mx.Config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
            raw_list = json.load(f)[: mx.Config.LIMIT_CONVERSATIONS]

        for data in raw_list:
            user_id = data.get("user_id")
            if user_id is None:
                mx.logger.warning("⚠️ Skipping entry without user_id")
                continue

            for qa_idx, qa in enumerate(data.get("qa", [])):
                if qa.get("category") == 5:
                    continue
                rankings, _, target_boxes = self._score_and_rank(user_id, qa, qa_idx)

                graph_info = None
                if self.graph_expand and rankings.get("content_event_topic_kw"):
                    ranked = rankings.get("content_event_topic_kw", [])
                    if self.top_k is not None and self.top_k > 0:
                        ranked = ranked[: self.top_k]
                    graph_info = self._expand_graph_for_blocks(user_id, ranked)

                writer.writerow(
                    [
                        user_id,
                        qa_idx,
                        qa.get("question", ""),
                        qa.get("category", ""),
                        rankings.get("content_event_topic_kw", []),
                        target_boxes,
                    ]
                )

                res_entry = {
                    "user_id": user_id,
                    "qa_idx": qa_idx,
                    "question": qa.get("question", ""),
                    "category": qa.get("category", ""),
                    "rankings": rankings,
                    "target_boxes": target_boxes,
                }
                if graph_info is not None:
                    res_entry["graph"] = graph_info
                mx.TraceLogger.log(result_jsonl, res_entry)

            mx.logger.info("✅ Simple retrieval done for user %s", user_id)

        csv_file.close()
        mx.logger.info("✅ Retrieval results appended to %s", result_jsonl)