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
        graph_include_relations: bool = True,
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
        self.graph_include_relations = bool(graph_include_relations)
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