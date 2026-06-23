from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
import time

def _mx():
    """
    Lazy import to avoid circular imports:
    memblock_extractor imports retrieval_impl, but retrieval_impl must not import memblock_extractor at import-time.
    """
    import memblock_extractor as mx  # local import by design

    return mx


class SimpleRetriever:
    """基于 content 向量与问题相似度的简易检索，不生成答案。"""

    def __init__(self, worker: Any, top_k: int = None):
        mx = _mx()
        self.worker = worker
        self.top_k = mx.Config.TOP_K_RETRIEVE if top_k is None else top_k
        self.all_boxes: List[Dict[str, Any]] = []
        self.trace_map: Dict[Any, Dict[int, List[int]]] = {}

        # Align lifecycle with enhanced retriever: one store per user in one run
        self.store_by_user: Dict[Any, Any] = {}
        self.query_count_by_user: Dict[Any, int] = {}
        self.flush_every_n_queries: int = 20

    def load(self):
        mx = _mx()
        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            self.all_boxes = [json.loads(l) for l in f if l.strip()]
        self.trace_map = self._load_traces(mx.Config.TIME_TRACE_FILE)

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
    def _get_or_create_store(self, user_id: Any):
        if user_id in self.store_by_user:
            return self.store_by_user[user_id], 0.0

        t0 = time.perf_counter()
        mx = _mx()
        store = mx.EmbeddingStore(self.worker, user_id)
        t_store_init = time.perf_counter() - t0
        self.store_by_user[user_id] = store
        return store, t_store_init


    def _flush_user_store(self, user_id: Any, force: bool = False) -> float:
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
                mx.logger.warning("flush failed for user %s: %s", uid, e)
        mx.logger.info("Final flush completed for %d user stores", flushed)

    def _score_and_rank(self, user_id: Any, qa: Dict[str, Any], qa_idx: int | None = None):
        mx = _mx()
        t_total_start = time.perf_counter()
        t_parse = 0.0
        t_filter = 0.0
        pool = [b for b in self.all_boxes if b.get("user_id") == user_id]
        q = qa.get("question", "") or ""
        q_id = qa.get("id", qa.get("question", ""))

        t_rank_start = time.perf_counter()

        # Reuse store per user (fair with enhanced)
        store, _ = self._get_or_create_store(user_id)

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
        rankings = {"content_event_topic_kw": ranked}

        # Periodic flush, not per-query flush
        self.query_count_by_user[user_id] = self.query_count_by_user.get(user_id, 0) + 1
        do_periodic_flush = (
            self.flush_every_n_queries is not None
            and self.flush_every_n_queries > 0
            and (self.query_count_by_user[user_id] % self.flush_every_n_queries == 0)
        )
        if do_periodic_flush:
            self._flush_user_store(user_id, force=False)

        t_rank = time.perf_counter() - t_rank_start
        t_total = time.perf_counter() - t_total_start

        # Keep baseline total timing for p50/p95
        mx.logger.info(
            "⏱️ [Baseline] Timing: q_idx=%s parse=%.6fs, filter=%.6fs, rank=%.6fs, total=%.3fs, pool=%d",
            str(qa_idx) if qa_idx is not None else "NA",
            t_parse,
            t_filter,
            t_rank,
            t_total,
            len(pool),
        )

        # Keep D-loop profile for core retrieval comparison
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
                mx.TraceLogger.log(result_jsonl, res_entry)

            mx.logger.info("✅ Simple retrieval done for user %s", user_id)
        self._flush_all_stores()
        csv_file.close()
        mx.logger.info("✅ Retrieval results appended to %s", result_jsonl)