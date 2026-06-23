from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity


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
        graph_expand: bool | None = None,
        graph_min_score: float | None = None,
        graph_limit: int | None = None,
        graph_hops: int | None = None,
        graph_include_relations: bool | None = None,
        graph_person_relations: bool | None = None,
    ):
        mx = _mx()
        self.worker = worker
        # top_k=None means keep full ranking; otherwise cap at provided value
        self.top_k = mx.Config.TOP_K_RETRIEVE if top_k is None else top_k
        self.all_boxes: List[Dict[str, Any]] = []
        self.trace_map: Dict[Any, Dict[int, List[int]]] = {}

        # Graph params are accepted for API compatibility but unused in locomo baseline.
        _ = (graph_expand, graph_min_score, graph_limit, graph_hops, graph_include_relations, graph_person_relations)

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

    def _score_and_rank(self, user_id: Any, qa: Dict[str, Any]):
        mx = _mx()
        pool = [b for b in self.all_boxes if b.get("user_id") == user_id]
        q = qa.get("question", "") or ""
        q_id = qa.get("id", qa.get("question", ""))

        store = mx.EmbeddingStore(self.worker, user_id)
        qvec = store.get_vector(f"qa_{user_id}_{q_id}", "question", q, note=f"U{user_id}_QA_Content")

        sim_map: Dict[int, float] = {}
        for b in pool:
            bid = mx._get_block_id(b)
            key = f"{user_id}_{bid}"

            # Get content_text (original conversation)
            content_text = b.get("features", {}).get("content_text", "") or ""

            # Get events (at top level, not in features)
            events = b.get("events", [])
            evt = mx._events_to_text(events)

            # Get topic keywords
            topic_kw = b.get("features", {}).get("topic_kw_text", "")

            # Combine all for retrieval ranking (Membox style)
            text = f"{content_text} {evt} {topic_kw}".strip()

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

        ranked = [bid for bid, _ in sorted(sim_map.items(), key=lambda x: x[1], reverse=True)]

        # Keep full ordering for downstream reuse; top-k slice is optional
        rankings = {"content_event_topic_kw": ranked}

        store.flush()
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

        # global_qa_idx = 0

        for conv_idx, data in enumerate(raw_list):
           
            user_id = str(conv_idx)  # Convert to string to match reindexed file format
            mx.logger.info(f"ℹ️ Assigned user_id={user_id} for conversation {conv_idx}")

            qa_count_in_conv = 0

            for qa_idx, qa in enumerate(data.get("qa", [])):
                if qa.get("category") == 5:
                    continue

                qa_count_in_conv += 1

                rankings, _, target_boxes = self._score_and_rank(user_id, qa)

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

            mx.logger.info(
                "✅ Simple retrieval done for user %s (conversation %d: qa_idx %d-%d, %d queries)",
                user_id, conv_idx, 0, qa_count_in_conv - 1, qa_count_in_conv
            )

        csv_file.close()
        mx.logger.info("✅ Retrieval results appended to %s", result_jsonl)