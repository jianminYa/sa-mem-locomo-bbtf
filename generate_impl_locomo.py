from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import tiktoken


def _mx():
    """
    Lazy import to avoid circular imports:
    memblock_extractor imports generate_impl, but generate_impl must not import memblock_extractor at import-time.
    """
    import memblock_extractor as mx  # local import by design

    return mx


class AnswerGenerator:
    """统一的生成与评估模块，可选问题重排。"""

    def __init__(
        self,
        worker: Any,
        answer_topn: int | List[int] | None = None,
        text_modes: List[str] | None = None,
        stage_label: str = "gen",
    ):
        mx = _mx()
        self.worker = worker
        if isinstance(answer_topn, list):
            self.answer_topn_list = answer_topn
        else:
            self.answer_topn_list = [answer_topn or mx.Config.ANSWER_TOP_N or mx.Config.TOP_K_RETRIEVE]
        self.text_modes = text_modes or mx.Config.GEN_TEXT_MODES
        self.trace_metrics = mx.Config.TRACE_METRICS
        try:
            self.encoding = tiktoken.encoding_for_model(mx.Config.LLM_MODEL)
        except KeyError:
            # Fallback for non-OpenAI model names.
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.box_map: Dict[Any, Dict[int, str]] = {}
        self.qa_map: Dict[Any, List[Dict[str, Any]]] = {}
        self.boxes_by_user: Dict[Any, List[Dict[str, Any]]] = {}
        self.trace_map: Dict[Any, Dict[str, List[Dict[str, Any]]]] = {}

        self.content_totals: Dict[Any, int] = defaultdict(int)

        self.aggregate: Dict[Tuple[str, str, str, str, int], Dict[str, float]] = defaultdict(
            lambda: {
                "f1_sum": 0.0,
                "bleu_sum": 0.0,
                "ctx_tokens_sum": 0.0,
                "generation_latency_sum": 0.0,
                "count": 0,
            }
        )
        self.aggregate_by_category: Dict[Tuple[str, str, str, str, int, str], Dict[str, float]] = defaultdict(
            lambda: {
                "f1_sum": 0.0,
                "bleu_sum": 0.0,
                "ctx_tokens_sum": 0.0,
                "generation_latency_sum": 0.0,
                "count": 0,
            }
        )

        self.conv_ctx_total: Dict[Tuple[str, Any], Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "count": 0.0})
        self.conv_ctx_by_mode: Dict[Tuple[str, Any, str], Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "count": 0.0})

        self.stage_label = stage_label

    @staticmethod
    def _tokens(text: str) -> List[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "").lower())
        return [t for t in cleaned.split() if t]

    @classmethod
    def _f1(cls, pred: str, gold: Any) -> float:
        pred_tokens = cls._tokens(pred)
        gold_list = gold if isinstance(gold, list) else [gold]
        best = 0.0
        for g in gold_list:
            gold_tokens = cls._tokens(g)
            if not gold_tokens or not pred_tokens:
                overlap = 0
            else:
                overlap = 0
                gold_counts: Dict[str, int] = {}
                for t in gold_tokens:
                    gold_counts[t] = gold_counts.get(t, 0) + 1
                for t in pred_tokens:
                    if t in gold_counts and gold_counts[t] > 0:
                        overlap += 1
                        gold_counts[t] -= 1
            if overlap == 0:
                f1 = 0.0
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)
            best = max(best, f1)
        return best

    @classmethod
    def _bleu(cls, pred: str, gold: Any) -> float:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        pred = str(pred)
        refs = [str(g) for g in gold] if isinstance(gold, list) else [str(gold)]
        pred_tokens = nltk.word_tokenize(pred.lower())
        refs_tokens = [nltk.word_tokenize(r.lower()) for r in refs]
        smooth = SmoothingFunction().method1
        try:
            return sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        except Exception:
            return 0.0

    def _load_boxes(self):
        mx = _mx()
        if not os.path.exists(mx.Config.FINAL_CONTENT_FILE):
            return
        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                b = json.loads(line)
                raw_sid = b.get("user_id")
                bid = mx._get_block_id(b)
                if raw_sid is None:
                    continue
                sid = str(raw_sid)
                # Get content_text (original conversation text)
                content_text = b.get("features", {}).get("content_text", "")

                # Store content_text for "content" mode (Membox style)
                # This will be used when mode == "content" in _generate_for_ranking
                if not hasattr(self, 'box_map_content'):
                    self.box_map_content = {}
                self.box_map_content.setdefault(sid, {})[bid] = content_text

                # Build event-based text for "event" mode (original SA-Mem style)
                events = b.get("events", [])
                topic_kw = b.get("features", {}).get("topic_kw_text", "")

                # Format events with temporal metadata
                event_descriptions = []
                for e in events:
                    desc = e.get("description", "")
                    if not desc:
                        continue

                    # Extract temporal metadata
                    time_meta = e.get("time_metadata", {})
                    start_time = time_meta.get("startTime")
                    end_time = time_meta.get("endTime")

                    # Format with temporal prefix
                    if start_time and end_time and start_time != end_time:
                        # Range: [2022-01-01 to 2022-12-31]
                        time_prefix = f"[{start_time} to {end_time}]"
                    elif start_time:
                        # Single date: [2023-05-07]
                        time_prefix = f"[{start_time}]"
                    else:
                        # No temporal metadata, use block-level fallback
                        block_time = b.get("temporal_index", {}).get("block_event_start_time")
                        time_prefix = f"[{block_time}]" if block_time else ""

                    # Combine time prefix with description
                    if time_prefix:
                        event_descriptions.append(f"{time_prefix} {desc}")
                    else:
                        event_descriptions.append(desc)

                # Combine topic keywords and event descriptions for "event" mode
                text_parts = []
                if topic_kw:
                    text_parts.append(f"Topics: {topic_kw}")
                if event_descriptions:
                    text_parts.append("Events:\n" + "\n".join(f"- {desc}" for desc in event_descriptions))

                event_text = "\n\n".join(text_parts) if text_parts else ""

                # Store event-based text for "event" mode
                if not hasattr(self, 'box_map_event'):
                    self.box_map_event = {}
                self.box_map_event.setdefault(sid, {})[bid] = event_text

                # Keep box_map for backward compatibility (defaults to content mode)
                self.box_map.setdefault(sid, {})[bid] = content_text

                self.boxes_by_user.setdefault(sid, []).append({"block_id": bid, "coverage": b.get("coverage", {})})

                # Use content_text for token counting (more accurate for content mode)
                self.content_totals[sid] += len(self.encoding.encode(content_text))

    def _load_qa(self):
        """
        Load QA pairs for LoCoMo with Membox-aligned indexing:
        - user_id is conversation index as string: "0", "1", "2", ...
        - qa_idx is local index within each conversation
        - category filtering is done in run(), not here
        """
        mx = _mx()
        if not os.path.exists(mx.Config.RAW_DATA_FILE):
            return

        with open(mx.Config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            limit = mx.Config.LIMIT_CONVERSATIONS
            raw_list = all_data if (limit is None or limit <= 0) else all_data[:limit]

        for conv_idx, data in enumerate(raw_list):
            user_id = str(conv_idx)
            qa_list = data.get("qa", []) or []
            self.qa_map[user_id] = qa_list

            valid_count = sum(1 for qa in qa_list if qa.get("category") != 5)
            mx.logger.info(
                f"ℹ️ Loaded {len(qa_list)} QA pairs for user_id={user_id} "
                f"(conversation {conv_idx}, non-cat5={valid_count})"
            )

    def _load_traces(self):
        mx = _mx()
        traces: Dict[Any, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        if os.path.exists(mx.Config.TIME_TRACE_FILE):
            with open(mx.Config.TIME_TRACE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    raw_sid = t.get("user_id")
                    metric = t.get("metric")
                    if raw_sid is None or metric is None:
                        continue
                    sid = str(raw_sid)
                    traces[sid][metric].append(t)
        self.trace_map = traces

    def _trace_events_for_box(self, sid: Any, bid: int, metric: str) -> List[str]:
        traces = self.trace_map.get(sid, {}).get(metric, [])
        for tr in traces:
            if bid not in tr.get("box_ids", []):
                continue
            events_texts: List[str] = []
            for entry in tr.get("entries", []):
                evs = entry.get("events") or []
                if not evs:
                    continue
                ts = str(entry.get("start_time", "Unknown"))
                for ev in evs:
                    ev_clean = str(ev).strip()
                    if ev_clean:
                        events_texts.append(f"{ts}: {ev_clean}")
            return events_texts
        return []

    def _build_trace_contexts(self, sid: Any, top_ids: List[int], trace_metric: str, mode: str) -> List[str]:
        contexts: List[str] = []
        seen_events = set()
        for bid in top_ids:
            events = self._trace_events_for_box(sid, bid, trace_metric)
            if not events:
                continue
            events_text = "\n".join(events)
            if events_text in seen_events:
                continue
            seen_events.add(events_text)
            if mode == "content_trace_event":
                content = self.box_map.get(sid, {}).get(bid)
                if not content:
                    continue
                contexts.append(f"{content}\nEvents:\n{events_text}")
            elif mode == "trace_event":
                contexts.append(f"Events:\n{events_text}")
        return contexts

    def _filter_expanded_events_for_context(self, expanded_events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        mx = _mx()
        topk = getattr(mx.Config, "GRAPH_CONTEXT_EXPANDED_TOPK", None)
        min_score = getattr(mx.Config, "GRAPH_CONTEXT_EXPANDED_MIN_SCORE", None)
        if min_score is None:
            min_score = 0.8

        prepared: List[Dict[str, Any]] = []
        for ev in expanded_events or []:
            if not isinstance(ev, dict):
                continue
            score_raw = ev.get("graph_similarity_score")
            score = None
            try:
                if score_raw is not None:
                    score = float(score_raw)
            except Exception:
                score = None
            copied = dict(ev)
            copied["graph_similarity_score"] = score
            prepared.append(copied)

        total_count = len(prepared)
        if min_score is not None:
            prepared = [
                ev for ev in prepared
                if ev.get("graph_similarity_score") is not None and float(ev.get("graph_similarity_score")) >= float(min_score)
            ]

        prepared.sort(
            key=lambda ev: (
                -(float(ev.get("graph_similarity_score")) if ev.get("graph_similarity_score") is not None else float("-inf")),
                str(ev.get("event_id") or ""),
            )
        )

        if topk is not None:
            prepared = prepared[: max(0, int(topk))]

        stats = {
            "expanded_events_total": total_count,
            "expanded_events_kept": len(prepared),
            "expanded_events_min_score": float(min_score) if min_score is not None else None,
            "expanded_events_topk": int(topk) if topk is not None else None,
        }
        return prepared, stats

    def _filter_person_relation_facts_for_context(
        self,
        person_relation_facts: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        mx = _mx()
        topk = getattr(mx.Config, "GRAPH_CONTEXT_EXPANDED_TOPK", None)
        min_score = getattr(mx.Config, "GRAPH_CONTEXT_EXPANDED_MIN_SCORE", None)
        if min_score is None:
            min_score = 0.5

        prepared: List[Dict[str, Any]] = []
        total_count = 0
        for item in person_relation_facts or []:
            person = str(item.get("person") or "").strip()
            facts = item.get("relation_facts") or []
            if not person or not isinstance(facts, list):
                continue
            copied_facts: List[Dict[str, Any]] = []
            for rel in facts:
                if not isinstance(rel, dict):
                    continue
                score_raw = rel.get("query_similarity")
                score = None
                try:
                    if score_raw is not None:
                        score = float(score_raw)
                except Exception:
                    score = None
                copied = dict(rel)
                copied["query_similarity"] = score
                copied_facts.append(copied)

            total_count += len(copied_facts)
            if min_score is not None:
                copied_facts = [
                    rel for rel in copied_facts
                    if rel.get("query_similarity") is not None
                    and float(rel.get("query_similarity")) >= float(min_score)
                ]

            copied_facts.sort(
                key=lambda rel: (
                    -(float(rel.get("query_similarity")) if rel.get("query_similarity") is not None else float("-inf")),
                    str(rel.get("relation_id") or ""),
                )
            )

            if topk is not None:
                copied_facts = copied_facts[: max(0, int(topk))]

            if copied_facts:
                prepared.append({
                    "person": person,
                    "relation_facts": copied_facts,
                })

        kept_count = sum(len(p.get("relation_facts") or []) for p in prepared)
        stats = {
            "person_relation_facts_total": total_count,
            "person_relation_facts_kept": kept_count,
            "person_relation_facts_min_score": float(min_score),
            "person_relation_facts_topk": int(topk) if topk is not None else None,
        }
        return prepared, stats

    @staticmethod
    def _format_time_range(start: Any, end: Any) -> str:
        start_s = str(start or "").strip()
        end_s = str(end or "").strip()
        if not start_s and not end_s:
            return ""
        if start_s and end_s and start_s != end_s:
            return f"{start_s} to {end_s}"
        return start_s or end_s

    def _format_graph_time_meta(self, item: Dict[str, Any]) -> str:
        event_time = self._format_time_range(item.get("event_start_time"), item.get("event_end_time"))
        block_time = self._format_time_range(item.get("block_event_start_time"), item.get("block_event_end_time"))
        source = str(item.get("time_source") or "").strip()

        meta: List[str] = []
        if event_time:
            meta.append(f"event_time={event_time}")
        elif block_time:
            meta.append(f"observed_time={block_time}")
        if source:
            meta.append(f"time_source={source}")
        return ", ".join(meta)

    def _format_relation_fact_text(self, rel: Dict[str, Any]) -> str:
        fact_text = str(rel.get("fact_text") or "").strip()
        if fact_text:
            return fact_text
        src = str(rel.get("source") or "").strip()
        rel_type = str(rel.get("relationship") or "").strip()
        dst = str(rel.get("destination") or "").strip()
        return f"{src} {rel_type} {dst}".strip()

    def _format_graph_context(self, graph_payload: Dict[str, Any], top_ids: List[int]) -> str:
        mx = _mx()
        if not graph_payload or not graph_payload.get("graph_enabled"):
            return ""
        events_only = bool(getattr(mx.Config, "GRAPH_CONTEXT_EVENTS_ONLY", False))
        relations_only = bool(getattr(mx.Config, "GRAPH_CONTEXT_RELATIONS_ONLY", False))
        use_person_profile = bool(getattr(mx.Config, "GRAPH_CONTEXT_PERSON_PROFILE", False))
        style = str(getattr(mx.Config, "GRAPH_CONTEXT_STYLE", "default") or "default").strip().lower()
        parts: List[str] = []

        if not relations_only:
            expanded_events = graph_payload.get("expanded_events") or []
            expanded_events, expanded_stats = self._filter_expanded_events_for_context(expanded_events)
            graph_payload["expanded_events_context_stats"] = expanded_stats
            if expanded_events:
                parts.append("Expanded events (similarity-filtered):")
                event_lines = []
                for ev in expanded_events:
                    desc = str(ev.get("event_description") or ev.get("event_id") or "").strip()
                    if not desc:
                        continue
                    score = ev.get("graph_similarity_score")
                    meta = []
                    if score is not None:
                        meta.append(f"graph_score={float(score):.4f}")
                    time_meta = self._format_graph_time_meta(ev)
                    if time_meta:
                        meta.append(time_meta)
                    prefix = f"[{', '.join(meta)}] " if meta else ""
                    event_lines.append(f"- {prefix}{desc}")
                if event_lines:
                    parts.extend(event_lines)

            # NOTE: Similar-event edges are intentionally omitted for now.

        if not events_only:
            relations = graph_payload.get("relations") or []
            if relations:
                parts.append("Relations:")
                if style == "fl-pro":
                    rel_lines = self._format_relation_lines_fl_pro(relations)
                else:
                    rel_lines = []
                    for rel in relations:
                        src = str(rel.get("source") or "").strip()
                        rel_type = str(rel.get("relationship") or "").strip()
                        dst = str(rel.get("destination") or "").strip()
                        if src and rel_type and dst:
                            time_meta = self._format_graph_time_meta(rel)
                            prefix = f"[{time_meta}] " if time_meta else ""
                            rel_lines.append(f"- {prefix}{src} -- {rel_type} -- {dst}")
                if rel_lines:
                    parts.extend(rel_lines)

        if use_person_profile:
            person_relation_facts = graph_payload.get("person_relation_facts") or []
            if person_relation_facts:
                filtered_profiles, profile_stats = self._filter_person_relation_facts_for_context(
                    person_relation_facts
                )
                graph_payload["person_relation_facts_context_stats"] = profile_stats
                if filtered_profiles:
                    parts.append("Person profiles:")
                for item in filtered_profiles:
                    person = str(item.get("person") or "").strip()
                    facts = item.get("relation_facts") or []
                    if not person or not facts:
                        continue
                    parts.append(f"About {person}, here is related history:")
                    for idx, rel in enumerate(facts, start=1):
                        fact_text = self._format_relation_fact_text(rel)
                        if fact_text:
                            time_meta = self._format_graph_time_meta(rel)
                            prefix = f"[{time_meta}] " if time_meta else ""
                            parts.append(f"{idx}. {prefix}{fact_text}")

        return "\n".join(parts)

    def _format_relation_lines_fl_pro(self, relations: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        for rel in relations:
            src = str(rel.get("source") or "").strip()
            rel_type = str(rel.get("relationship") or "").strip()
            dst = str(rel.get("destination") or "").strip()
            if not (src and rel_type and dst):
                continue
            rel_text = rel_type.replace("_", " ").strip()
            time_meta = self._format_graph_time_meta(rel)
            prefix = f"[{time_meta}] " if time_meta else ""
            lines.append(f"- {prefix}Relation between {src} and {dst}: {rel_text}.")
        return lines

    def _log_token_counts(self, context_text: str, question: str) -> Dict[str, Any]:
        mx = _mx()
        return {
            "memories_tokens": len(self.encoding.encode(context_text)),
            "question_tokens": len(self.encoding.encode(question)),
            "prompt_tokens_est": len(self.encoding.encode(mx.Config.PROMPT_QA_ANSWER.format(memories=context_text, question=question))),
        }

    def _record_metrics(
        self,
        ranking_strategy: str,
        metric: str,
        trace_metric: str | None,
        mode: str,
        topn: int,
        f1: float,
        bleu: float,
        ctx_tokens: int,
        generation_latency: float,
        sid: Any,
        category: Any,
    ):
        key = (ranking_strategy, metric, trace_metric or "", mode, topn)
        agg = self.aggregate[key]
        agg["f1_sum"] += f1
        agg["bleu_sum"] += bleu
        agg["ctx_tokens_sum"] += ctx_tokens
        agg["generation_latency_sum"] = agg.get("generation_latency_sum", 0.0) + generation_latency
        agg["count"] += 1

        cat_label = "uncategorized" if category is None else str(category)
        cat_key = (ranking_strategy, metric, trace_metric or "", mode, topn, cat_label)
        cat_agg = self.aggregate_by_category[cat_key]
        cat_agg["f1_sum"] += f1
        cat_agg["bleu_sum"] += bleu
        cat_agg["ctx_tokens_sum"] += ctx_tokens
        cat_agg["generation_latency_sum"] = cat_agg.get("generation_latency_sum", 0.0) + generation_latency
        cat_agg["count"] += 1

        conv_key = (ranking_strategy, sid)
        conv_stat_total = self.conv_ctx_total[conv_key]
        conv_stat_total["tokens"] += ctx_tokens
        conv_stat_total["count"] += 1

        conv_mode_key = (ranking_strategy, sid, mode)
        conv_stat_mode = self.conv_ctx_by_mode[conv_mode_key]
        conv_stat_mode["tokens"] += ctx_tokens
        conv_stat_mode["count"] += 1

    def _write_summary(self):
        mx = _mx()
        if not self.aggregate:
            return

        records = []
        for (ranking_strategy, metric, trace_metric, mode, topn), v in self.aggregate.items():
            count = max(v.get("count", 0), 1)
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "metric": metric,
                    "trace_metric": trace_metric,
                    "text_mode": mode,
                    "topn": topn,
                    "avg_f1": round(v.get("f1_sum", 0) / count, 4),
                    "avg_bleu": round(v.get("bleu_sum", 0) / count, 4),
                    "avg_context_tokens": round(v.get("ctx_tokens_sum", 0) / count, 2),
                    "avg_generation_latency": round(v.get("generation_latency_sum", 0.0) / count, 4),
                    "count": v.get("count", 0),
                }
            )

        for (ranking_strategy, metric, trace_metric, mode, topn, category), v in self.aggregate_by_category.items():
            count = max(v.get("count", 0), 1)
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "metric": metric,
                    "trace_metric": trace_metric,
                    "text_mode": mode,
                    "topn": topn,
                    "category": category,
                    "avg_f1": round(v.get("f1_sum", 0) / count, 4),
                    "avg_bleu": round(v.get("bleu_sum", 0) / count, 4),
                    "avg_context_tokens": round(v.get("ctx_tokens_sum", 0) / count, 2),
                    "avg_generation_latency": round(v.get("generation_latency_sum", 0.0) / count, 4),
                    "count": v.get("count", 0),
                    "type": "category_metrics",
                }
            )

        for (ranking_strategy, sid), stat in self.conv_ctx_total.items():
            count = max(stat.get("count", 0), 1)
            avg_ctx = stat.get("tokens", 0) / count
            content_total = self.content_totals.get(sid, 1)
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "user_id": sid,
                    "avg_context_tokens": round(avg_ctx, 2),
                    "content_tokens_total": content_total,
                    "avg_context_ratio_over_content": round(avg_ctx / max(content_total, 1), 4),
                    "count": stat.get("count", 0),
                    "type": "conversation_context_usage",
                }
            )

        for (ranking_strategy, sid, mode), stat in self.conv_ctx_by_mode.items():
            content_total = self.content_totals.get(sid, 1)
            count = max(stat.get("count", 0), 1)
            avg_ctx = stat.get("tokens", 0) / count
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "user_id": sid,
                    "text_mode": mode,
                    "avg_context_tokens": round(avg_ctx, 2),
                    "content_tokens_total": content_total,
                    "avg_context_ratio_over_content": round(avg_ctx / max(content_total, 1), 4),
                    "count": stat.get("count", 0),
                    "type": "conversation_context_usage_by_mode",
                }
            )

        os.makedirs(os.path.dirname(mx.Config.GEN_SUMMARY_FILE), exist_ok=True)
        with open(mx.Config.GEN_SUMMARY_FILE, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _generate_for_ranking(
        self,
        *,
        ranking_strategy: str,
        metric: str,
        trace_metric: str | None,
        mode: str,
        top_ids: List[int],
        sid: Any,
        qid: int,
        question: str,
        gold: Any,
        targets: List[int],
        category: Any,
        writer: csv.writer,
        out_jsonl: str,
        topn: int,
        graph_payload: Dict[str, Any] | None = None,
    ):
        mx = _mx()

        # Select appropriate box_map based on mode
        if mode == "content":
            # Membox style: Use only original conversation text (content_text)
            box_map = getattr(self, 'box_map_content', self.box_map)
            contexts = [box_map.get(sid, {}).get(bid) for bid in top_ids if box_map.get(sid, {}).get(bid)]
        elif mode == "event":
            # SA-Mem style: Use events with temporal metadata
            box_map = getattr(self, 'box_map_event', {})
            contexts = [box_map.get(sid, {}).get(bid) for bid in top_ids if box_map.get(sid, {}).get(bid)]
        else:
            # Trace modes (content_trace_event, trace_event)
            contexts = self._build_trace_contexts(sid, top_ids, trace_metric or "content_event_topic_kw", mode)

        if not contexts:
            return

        context_text = "\n\n".join(contexts)
        use_graph = bool(getattr(mx.Config, "USE_GRAPH_CONTEXT", False))
        allowed_categories = getattr(mx.Config, "GRAPH_CONTEXT_CATEGORIES", None)
        if use_graph and allowed_categories is not None:
            try:
                cat_int = int(category) if category is not None else None
            except Exception:
                cat_int = None
            use_graph = bool(cat_int is not None and cat_int in set(allowed_categories))

        graph_text = self._format_graph_context(graph_payload or {}, top_ids) if use_graph else ""
        if graph_text:
            graph_intro = (
                "Supplementary Graph Evidence:\n"
                "Use the following graph-expanded events, relation facts, and person profiles only as supporting evidence. "
                "Prefer direct retrieved memories above, and ignore graph items that are irrelevant, weakly related, "
                "or contradicted by the main memories.\n"
            )
            context_text = f"{context_text}\n\n{graph_intro}{graph_text}"
        user_prompt = mx.Config.PROMPT_QA_ANSWER.format(memories=context_text, question=question)
        note = f"S{sid}_QA_{qid}_{ranking_strategy}_{metric}_top{topn}_{trace_metric or 'content'}_{mode}"
        token_info = self._log_token_counts(context_text, question)

        generation_start = time.perf_counter()
        ans = self.worker.chat_completion(
            user_prompt,
            note=note,
            extra={**token_info, "stage": f"{self.stage_label}:{ranking_strategy}"},
        )
        generation_latency = time.perf_counter() - generation_start

        f1 = self._f1(ans, gold)
        bleu = self._bleu(ans, gold)
        ctx_tokens = int(token_info.get("memories_tokens", 0) or 0)

        writer.writerow(
            [
                sid,
                qid,
                ranking_strategy,
                question,
                gold,
                ans,
                f"{f1:.4f}",
                f"{bleu:.4f}",
                metric,
                trace_metric or "",
                mode,
                topn,
                top_ids,
                targets,
                category,
                ctx_tokens,
                f"{generation_latency:.6f}",
            ]
        )

        mx.TraceLogger.log(
            out_jsonl,
            {
                "user_id": sid,
                "qa_idx": qid,
                "ranking_strategy": ranking_strategy,
                "question": question,
                "gold": gold,
                "pred": ans,
                "f1": f1,
                "bleu": bleu,
                "metric": metric,
                "trace_metric": trace_metric,
                "text_mode": mode,
                "topn": topn,
                "topk": top_ids,
                "target_boxes": targets,
                "category": category,
                "context_tokens": ctx_tokens,
                "generation_latency": generation_latency,
                "graph_context_used": bool(graph_text),
                "graph_context_categories": sorted(list(allowed_categories)) if allowed_categories is not None else None,
                "graph_payload_enabled": bool((graph_payload or {}).get("graph_enabled")),
            },
        )

        self._record_metrics(
            ranking_strategy,
            metric,
            trace_metric,
            mode,
            topn,
            f1,
            bleu,
            ctx_tokens,
            generation_latency,
            sid,
            category,
        )

    def run(self, retrieval_jsonl: str, base_out_jsonl: str, base_out_csv: str):
        mx = _mx()
        if not os.path.exists(retrieval_jsonl):
            mx.logger.error("❌ Retrieval result not found: %s", retrieval_jsonl)
            return

        self._load_boxes()
        self._load_qa()
        self._load_traces()

        mx.logger.info("ℹ️ Generation text_modes=%s answer_topn=%s", self.text_modes, self.answer_topn_list)

        csv_base_exists = os.path.exists(base_out_csv)
        os.makedirs(os.path.dirname(base_out_csv), exist_ok=True)
        csv_base_file = open(base_out_csv, "a", newline="", encoding="utf-8")
        base_writer = csv.writer(csv_base_file)
        if not csv_base_exists:
            base_writer.writerow(
                [
                    "User_ID",
                    "QA_ID",
                    "Ranking_Strategy",
                    "Question",
                    "Gold",
                    "Pred",
                    "F1",
                    "BLEU",
                    "Metric",
                    "Trace_Metric",
                    "Text_Mode",
                    "TopN",
                    "TopIDs",
                    "Targets",
                    "Category",
                    "Context_Tokens",
                    "Generation_Latency",
                ]
            )

        with open(retrieval_jsonl, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        for ent in entries:
            sid_raw = ent.get("user_id")
            qid_raw = ent.get("qa_idx")
            if sid_raw is None or qid_raw is None:
                continue
            sid = str(sid_raw)
            try:
                qid = int(qid_raw)
            except Exception:
                continue
            qa_list = self.qa_map.get(sid, [])
            if qid<0 or qid >= len(qa_list):
                continue
            qa = qa_list[qid]
            question = qa.get("question", "")
            gold = qa.get("answer", "")
            category = qa.get("category")
            if category == 5:
                continue

            targets = mx.evidence_to_targets(qa.get("evidence"), self.boxes_by_user.get(sid, []))

            rankings = ent.get("rankings", {}) or {}
            base_rank = rankings.get("content_event_topic_kw", []) or []
            if not base_rank:
                continue

            ranking_sets = [("baseline", base_rank, base_writer, base_out_jsonl)]
            graph_payload = ent.get("graph") or {}

            for ranking_strategy, ranking_list, writer_obj, out_path in ranking_sets:
                if not writer_obj or not out_path:
                    continue

                for topn in self.answer_topn_list:
                    top_ids = ranking_list[: topn]
                    if not top_ids:
                        continue

                    # Content mode: Membox style (only original conversation text)
                    if "content" in self.text_modes:
                        self._generate_for_ranking(
                            ranking_strategy=ranking_strategy,
                            metric="content_event_topic_kw",
                            trace_metric=None,
                            mode="content",
                            top_ids=top_ids,
                            sid=sid,
                            qid=qid,
                            question=question,
                            gold=gold,
                            targets=targets,
                            category=category,
                            writer=writer_obj,
                            out_jsonl=out_path,
                            topn=topn,
                            graph_payload=graph_payload,
                        )

                    # Event mode: SA-Mem style (events with temporal metadata)
                    if "event" in self.text_modes:
                        self._generate_for_ranking(
                            ranking_strategy=ranking_strategy,
                            metric="content_event_topic_kw",
                            trace_metric=None,
                            mode="event",
                            top_ids=top_ids,
                            sid=sid,
                            qid=qid,
                            question=question,
                            gold=gold,
                            targets=targets,
                            category=category,
                            writer=writer_obj,
                            out_jsonl=out_path,
                            topn=topn,
                            graph_payload=graph_payload,
                        )

                    if "content_trace_event" in self.text_modes or "trace_event" in self.text_modes:
                        for trace_metric in self.trace_metrics:
                            for mode in [m for m in self.text_modes if m in ("content_trace_event", "trace_event")]:
                                self._generate_for_ranking(
                                    ranking_strategy=ranking_strategy,
                                    metric="content_event_topic_kw",
                                    trace_metric=trace_metric,
                                    mode=mode,
                                    top_ids=top_ids,
                                    sid=sid,
                                    qid=qid,
                                    question=question,
                                    gold=gold,
                                    targets=targets,
                                    category=category,
                                    writer=writer_obj,
                                    out_jsonl=out_path,
                                    topn=topn,
                                    graph_payload=graph_payload,
                                )

        csv_base_file.close()
        self._write_summary()
        mx.logger.info("✅ Generation complete")
