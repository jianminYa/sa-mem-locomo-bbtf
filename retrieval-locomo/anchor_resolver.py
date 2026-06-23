from __future__ import annotations

"""
Anchor Query Resolver for complex temporal queries.

Handles queries like:
- "Was I married when I lived in Paris?"
- "What did I do after I graduated?"
- "Where did I work before moving to NYC?"

Two-stage approach:
1. Find the anchor event (e.g., "lived in Paris", "graduated")
2. Query within the anchor's time range
"""


'''
  Confidence gating (in _find_anchor_blocks) — DUAL-CHANNEL ACCEPTANCE.
  Accept the anchor if EITHER channel passes; otherwise fallback to full pool.

  ┌─────────────────────────┬──────┬─────────────────────────────────────────┐
  │   通用                   │       │                                         │
  ├─────────────────────────┼──────┼─────────────────────────────────────────┤
  │ SCORE_THRESHOLD          │ 0.4  │ 单块入选 top_blocks 的最低 cosine        │
  └─────────────────────────┴──────┴─────────────────────────────────────────┘

  Channel A — CLUSTER（多匹配集群）所有条件都要满足:
  ┌─────────────────────────┬──────┬─────────────────────────────────────────┐
  │ CLUSTER_MIN_MATCHES      │ 2    │ 至少几个候选过 SCORE_THRESHOLD            │
  │ CLUSTER_MIN_TOP1         │ 0.46 │ top-1 cosine 最低线                       │
  │ CLUSTER_MIN_MARGIN       │ 0.05 │ top1 vs median 候选差距，过滤"一片差不多" │
  │ CLUSTER_MIN_AVG_TOP3     │ 0.45 │ top-3 平均，确保 union 在真正的簇上        │
  └─────────────────────────┴──────┴─────────────────────────────────────────┘

  Channel B — LONE_PEAK（强孤峰）所有条件都要满足:
  ┌─────────────────────────────┬──────┬─────────────────────────────────────┐
  │ LONE_PEAK_MIN_TOP1            │ 0.5  │ top-1 cosine 必须够强                  │
  │ LONE_PEAK_MIN_RUNNERUP_MARGIN │ 0.05 │ top1 - top2（全榜，含未过 0.4 的）     │
  └─────────────────────────────┴──────┴─────────────────────────────────────┘
  LONE_PEAK 通过时只保留 top-1，不做 union——单一可信锚点优于污染的 union。
'''


from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import re


def _mx():
    """Lazy import to avoid circular imports."""
    import memblock_extractor as mx
    return mx


class AnchorResolver:
    """Resolves anchor-based temporal queries."""

    def __init__(self, temporal_index, worker: Any):
        """
        Args:
            temporal_index: TemporalIndex instance
            worker: LLMWorker instance for embeddings
        """
        self.temporal_index = temporal_index
        self.worker = worker
        # Populated by resolve_anchor(); holds the anchor blocks found in the
        # last call so that callers can merge them back into the filtered pool
        # (e.g. for temporal-reasoning queries that need both sides).
        self.last_anchor_blocks: List[Dict[str, Any]] = []

    def resolve_anchor(
        self,
        anchor_text: str,
        anchor_relation: str,
        user_id: Any,
        pool: List[Dict[str, Any]]
    ) -> Tuple[Optional[date], Optional[date], str]:
        """
        Resolve an anchor event to a time range plus the temporal query type.

        Strategy:
          - Find up to top-10 anchor candidate blocks (or all blocks if pool < 10).
          - Treat each candidate as an independent anchor hypothesis and compute
            its own (start, end) per `anchor_relation`.
          - Take the UNION of all candidate windows so we do not aggressively
            filter out the true anchor when embedding matching picks a
            semantically similar but wrong block.

        Args:
            anchor_text: The anchor event description (e.g., "when I lived in Paris")
            anchor_relation: Temporal relation ("DURING", "BEFORE", "AFTER")
            user_id: User identifier
            pool: Pool of candidate blocks

        Returns:
            Tuple of (ref_start, ref_end, time_type):
              - DURING  : (min_start, max_end, "RANGE")   — union of per-candidate ranges
              - BEFORE  : (max_start, None,    "BEFORE")  — least restrictive cutoff
              - AFTER   : (min_end,   None,    "AFTER")   — least restrictive cutoff
              - failure : (None, None, "NONE")
        """
        mx = _mx()
        import time

        # Step 1: Find blocks that match the anchor event using vector similarity.
        # Use up to top-10 candidates (or the entire pool if smaller).
        t_find_start = time.perf_counter()
        candidate_top_k = min(10, len(pool)) if pool else 0
        anchor_blocks = self._find_anchor_blocks(
            anchor_text, user_id, pool, top_k=candidate_top_k
        )
        t_find = time.perf_counter() - t_find_start

        # Expose anchor blocks for callers that need to merge them back.
        self.last_anchor_blocks = anchor_blocks if anchor_blocks else []

        if not anchor_blocks:
            mx.logger.warning("⚠️ No blocks found for anchor: '%s'", anchor_text)
            return None, None, "NONE"

        # Step 2: For EACH anchor candidate, compute its own time window and
        # take the union. This guards against a single wrong embedding match
        # hard-filtering out the true target.
        t_extract_start = time.perf_counter()
        per_candidate_windows: List[Tuple[date, date]] = []
        for blk in anchor_blocks:
            s, e = self._extract_time_range([blk])
            if s and e:
                per_candidate_windows.append((s, e))
        t_extract = time.perf_counter() - t_extract_start

        if not per_candidate_windows:
            mx.logger.warning("⚠️ Could not extract time range from any anchor candidate")
            return None, None, "NONE"

        # Step 3: Combine per-candidate windows into a single (ref_start, ref_end, time_type)
        # tuple consumable by TemporalIndex.query_temporal. The union semantics
        # depend on the temporal relation.
        rel = (anchor_relation or "DURING").upper()
        starts = [w[0] for w in per_candidate_windows]
        ends = [w[1] for w in per_candidate_windows]

        if rel == "BEFORE":
            # BEFORE x: keep blocks with event_time < x. Union over candidates =>
            # use the LATEST candidate's start as the cutoff (least restrictive).
            ref_start, ref_end, time_type = max(starts), None, "BEFORE"
        elif rel == "AFTER":
            # AFTER x: keep blocks with event_time > x. Union over candidates =>
            # use the EARLIEST candidate's end as the cutoff (least restrictive).
            ref_start, ref_end, time_type = min(ends), None, "AFTER"
        else:
            # DURING / unknown: union of closed ranges => [min_start, max_end].
            ref_start, ref_end, time_type = min(starts), max(ends), "RANGE"

        mx.logger.info(
            "✅ Anchor resolved (union of %d candidates): '%s' %s -> type=%s start=%s end=%s",
            len(per_candidate_windows), anchor_text, anchor_relation, time_type, ref_start, ref_end,
        )
        mx.logger.info(
            "⏱️ Anchor resolution timing: find_blocks=%.3fs, extract_time=%.3fs, total=%.3fs",
            t_find, t_extract, t_find + t_extract
        )

        return ref_start, ref_end, time_type

    def _find_anchor_blocks(
        self,
        anchor_text: str,
        user_id: Any,
        pool: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find blocks that best match the anchor event description.

        Uses vector similarity to find relevant blocks. Returns up to ``top_k``
        candidates (or all blocks above the score threshold if pool is smaller).
        The caller treats these as independent anchor hypotheses and unions
        their time windows, so picking a moderately larger top_k here gives a
        wider safety net against single embedding-matching errors.
        """
        mx = _mx()

        # Get embedding for anchor text
        store = mx.EmbeddingStore(self.worker, user_id)
        anchor_vec = store.get_vector(
            f"anchor_{user_id}_{hash(anchor_text)}",
            "anchor_query",
            anchor_text,
            note=f"U{user_id}_Anchor"
        )

        from sklearn.metrics.pairwise import cosine_similarity
        import time as _time_mod

        # ── Stage 1: Block-level coarse filtering ──────────────────────
        # Reuse cached block-level embeddings (content_event_topic_kw) to
        # quickly narrow down candidates. All embeddings should already be
        # cached from the similarity ranking stage, so this is zero API calls.
        COARSE_TOP_N = 30
        t_coarse_start = _time_mod.perf_counter()

        block_coarse_scores = []
        for block in pool:
            block_id = mx._get_block_id(block)
            key = f"{user_id}_{block_id}"
            features = block.get("features", {})
            topic_kw = features.get("topic_kw_text", "")
            events = block.get("events", [])
            event_texts = [e.get("description", "") for e in events if e.get("description")]
            event_str = " | ".join(event_texts[:20])
            text = f"{topic_kw} {event_str}".strip()
            block_vec = store.get_vector(
                key, "content_event_topic_kw", text,
                note=f"U{user_id}_B{block_id}_anchor_coarse",
            )
            try:
                sim = cosine_similarity([anchor_vec], [block_vec])[0][0] if block_vec else -1.0
            except Exception:
                sim = -1.0
            block_coarse_scores.append((block, float(sim)))

        block_coarse_scores.sort(key=lambda x: x[1], reverse=True)
        coarse_pool = [b for b, _ in block_coarse_scores[:COARSE_TOP_N]]
        t_coarse = _time_mod.perf_counter() - t_coarse_start

        mx.logger.info(
            "🔍 Anchor Stage1 (block coarse): top-%d/%d blocks in %.3fs "
            "(top1=%.3f, top30=%.3f)",
            COARSE_TOP_N, len(pool), t_coarse,
            block_coarse_scores[0][1] if block_coarse_scores else -1.0,
            block_coarse_scores[min(COARSE_TOP_N - 1, len(block_coarse_scores) - 1)][1]
            if block_coarse_scores else -1.0,
        )

        # ── Stage 2: Event-level fine-grained matching ─────────────────
        # Only process events from the coarse-filtered blocks (~150 events
        # instead of ~700), drastically reducing embedding API calls.
        t_fine_start = _time_mod.perf_counter()

        event_scores = []  # List[(block, event_idx, sim)]
        for block in coarse_pool:
            block_id = mx._get_block_id(block)
            events = block.get("events", [])
            for idx, evt in enumerate(events):
                desc = evt.get("description", "")
                if not desc:
                    continue
                key = f"{user_id}_{block_id}_e{idx}"
                evt_vec = store.get_vector(
                    key, "event_description", desc,
                    note=f"U{user_id}_B{block_id}_e{idx}_anchor",
                )
                try:
                    sim = cosine_similarity([anchor_vec], [evt_vec])[0][0] if evt_vec else -1.0
                except Exception:
                    sim = -1.0
                event_scores.append((block, idx, float(sim)))

        # Sort by event-level similarity (descending)
        event_scores.sort(key=lambda x: x[2], reverse=True)

        # Aggregate to block level: each block takes its best event's score.
        seen_block_ids = {}  # block_id -> (block, best_sim)
        for block, idx, sim in event_scores:
            bid = mx._get_block_id(block)
            if bid not in seen_block_ids:
                seen_block_ids[bid] = (block, sim)

        scores = [(blk, sim) for blk, sim in seen_block_ids.values()]
        scores.sort(key=lambda x: x[1], reverse=True)

        t_fine = _time_mod.perf_counter() - t_fine_start
        mx.logger.info(
            "🔍 Anchor Stage2 (event fine): %d events from %d blocks in %.3fs",
            len(event_scores), len(coarse_pool), t_fine,
        )

        # Per-block inclusion threshold. Calibrated to the embedding model's
        # actual similarity distribution on this dataset (top-1 cosines for
        # genuinely matching anchors observed in the 0.50~0.60 band, weak/no
        # matches concentrate below 0.40). 0.40 keeps real anchor candidates
        # while still filtering the long tail of near-noise matches.
        SCORE_THRESHOLD = 0.4
        top_blocks = [block for block, score in scores[:top_k] if score > SCORE_THRESHOLD]

        # ────────────────────────────────────────────────────────────────────
        # Confidence gating
        # ────────────────────────────────────────────────────────────────────
        # We empirically observed that "rescuing" anchors via a LONE_PEAK
        # channel (top1 strong but only one match above SCORE_THRESHOLD) HURT
        # accuracy: with only one candidate the AFTER/BEFORE cutoff is taken
        # directly off the anchor block itself, and the index's strict
        # inequality (>=, <=) then excludes the anchor block — which is often
        # also the answer block. The CLUSTER channel needs ≥2 matches, so the
        # cutoff is shifted away from any single block and this issue does
        # not occur.
        #
        # We therefore keep the LONE_PEAK code path but disable it by default;
        # toggle ENABLE_LONE_PEAK to re-enable for experimentation.
        ENABLE_LONE_PEAK = False

        # Channel A (CLUSTER) gates:
        CLUSTER_MIN_MATCHES = 1
        CLUSTER_MIN_TOP1 = 0.5
        CLUSTER_MIN_MARGIN = 0.05         # top1 vs median of surviving
        CLUSTER_MIN_AVG_TOP3 = 0.45
        # Channel B (LONE_PEAK) gates (only used if ENABLE_LONE_PEAK):
        LONE_PEAK_MIN_TOP1 = 0.5
        LONE_PEAK_MIN_RUNNERUP_MARGIN = 0.05

        top_scores = [s for _, s in scores[:top_k] if s > SCORE_THRESHOLD]
        top1_sim = top_scores[0] if top_scores else -1.0

        # Top-2 over the FULL ranking (used by the LONE_PEAK channel).
        all_top1 = scores[0][1] if scores else -1.0
        all_top2 = scores[1][1] if len(scores) > 1 else -1.0
        runnerup_margin = all_top1 - all_top2

        # Median of surviving candidates (robust against outliers)
        if top_scores:
            sorted_scores = sorted(top_scores)
            mid = len(sorted_scores) // 2
            median_sim = (
                sorted_scores[mid]
                if len(sorted_scores) % 2 == 1
                else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
            )
        else:
            median_sim = -1.0

        avg_top3 = (
            sum(top_scores[:3]) / min(3, len(top_scores)) if top_scores else -1.0
        )

        # Channel A: CLUSTER acceptance
        cluster_ok = (
            len(top_blocks) >= CLUSTER_MIN_MATCHES
            and top1_sim >= CLUSTER_MIN_TOP1
            and (top1_sim - median_sim) >= CLUSTER_MIN_MARGIN
            and avg_top3 >= CLUSTER_MIN_AVG_TOP3
        )
        # Channel B: LONE_PEAK acceptance (gated by ENABLE_LONE_PEAK)
        lone_peak_ok = ENABLE_LONE_PEAK and (
            len(top_blocks) >= 1
            and top1_sim >= LONE_PEAK_MIN_TOP1
            and runnerup_margin >= LONE_PEAK_MIN_RUNNERUP_MARGIN
        )

        if not (cluster_ok or lone_peak_ok):
            # Build a human-readable rejection reason.
            reasons = []
            if not cluster_ok:
                if len(top_blocks) < CLUSTER_MIN_MATCHES:
                    reasons.append(f"cluster:matches={len(top_blocks)}<{CLUSTER_MIN_MATCHES}")
                elif top1_sim < CLUSTER_MIN_TOP1:
                    reasons.append(f"cluster:top1={top1_sim:.3f}<{CLUSTER_MIN_TOP1}")
                elif (top1_sim - median_sim) < CLUSTER_MIN_MARGIN:
                    reasons.append(
                        f"cluster:top1-median={top1_sim - median_sim:.3f}<{CLUSTER_MIN_MARGIN}"
                    )
                elif avg_top3 < CLUSTER_MIN_AVG_TOP3:
                    reasons.append(f"cluster:avg_top3={avg_top3:.3f}<{CLUSTER_MIN_AVG_TOP3}")
            if not lone_peak_ok:
                if top1_sim < LONE_PEAK_MIN_TOP1:
                    reasons.append(f"lone:top1={top1_sim:.3f}<{LONE_PEAK_MIN_TOP1}")
                elif runnerup_margin < LONE_PEAK_MIN_RUNNERUP_MARGIN:
                    reasons.append(
                        f"lone:top1-top2={runnerup_margin:.3f}<{LONE_PEAK_MIN_RUNNERUP_MARGIN}"
                    )
            mx.logger.warning(
                "⚠️ Anchor '%s' low confidence (%s | matches=%d, top1=%.3f, "
                "top2=%.3f, median=%.3f, avg_top3=%.3f) -> fallback",
                anchor_text[:60], "; ".join(reasons), len(top_blocks),
                top1_sim, all_top2, median_sim, avg_top3,
            )
            store.flush()
            return []

        # If only LONE_PEAK passes, restrict candidates to the single strong
        # peak — a single trustworthy anchor is better than a polluted union.
        if lone_peak_ok and not cluster_ok:
            top_blocks = top_blocks[:1]
            mx.logger.info(
                "✓ Accepted via LONE_PEAK channel (top1=%.3f, top1-top2=%.3f) "
                "→ keeping single strongest candidate",
                top1_sim, runnerup_margin,
            )

        store.flush()

        # Log the best matching event for debugging
        best_event_desc = ""
        if event_scores:
            best_block, best_idx, best_sim = event_scores[0]
            best_events = best_block.get("events", [])
            if best_idx < len(best_events):
                best_event_desc = best_events[best_idx].get("description", "")[:60]
        mx.logger.info(
            "🔍 Anchor resolution (event-level): found %d/%d blocks matching '%s' "
            "(top1=%.3f, median=%.3f, avg_top3=%.3f, processed %d blocks / %d events) "
            "best_event='%s'",
            len(top_blocks), top_k, anchor_text[:30],
            top1_sim, median_sim, avg_top3, len(pool), len(event_scores),
            best_event_desc,
        )

        return top_blocks

    def _extract_time_range(
        self,
        blocks: List[Dict[str, Any]]
    ) -> Tuple[Optional[date], Optional[date]]:
        """
        Extract the time range covered by a set of blocks.

        Uses event_time only. Blocks missing event_time are skipped rather
        than falling back to session_time — session_time reflects when the
        conversation happened, not when the anchor event actually occurred,
        and using it pollutes the inferred anchor window with unrelated
        conversation dates. If no block contributes an event_time, returns
        (None, None) so the caller can fall back to the full pool.
        """
        start_dates: List[date] = []
        end_dates: List[date] = []
        skipped = 0

        for block in blocks:
            temporal_idx = block.get("temporal_index", {})

            event_start = self._parse_date(temporal_idx.get("block_event_start_time"))
            event_end = self._parse_date(temporal_idx.get("block_event_end_time"))

            if event_start and event_end:
                start_dates.append(event_start)
                end_dates.append(event_end)
            else:
                skipped += 1

        if skipped:
            _mx().logger.info(
                "ℹ️ _extract_time_range skipped %d/%d anchor blocks lacking event_time",
                skipped, len(blocks),
            )

        if not start_dates or not end_dates:
            return None, None

        return min(start_dates), max(end_dates)

    @staticmethod
    def _parse_date(date_str: str) -> Optional[date]:
        """Parse ISO date string to date object."""
        if not date_str or date_str == "Unknown":
            return None

        try:
            if "T" in date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                return dt.date()
            else:
                return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return None

    @staticmethod
    def _apply_relation(
        anchor_start: date,
        anchor_end: date,
        relation: str
    ) -> Tuple[Optional[date], Optional[date], str]:
        """
        Apply temporal relation to anchor time range.

        Returns:
            (ref_start, ref_end, time_type) consumable by
            TemporalIndex.query_temporal:
              - DURING  -> (anchor_start, anchor_end, "RANGE")
              - BEFORE  -> (anchor_start, None,       "BEFORE")
                           (query_before uses start as the cutoff; -inf on the left)
              - AFTER   -> (anchor_end,   None,       "AFTER")
                           (query_after uses start as the cutoff; +inf on the right)
        """
        rel = (relation or "DURING").upper()

        if rel == "BEFORE":
            return anchor_start, None, "BEFORE"

        if rel == "AFTER":
            return anchor_end, None, "AFTER"

        # DURING and any unknown relation default to closed range.
        return anchor_start, anchor_end, "RANGE"


def resolve_anchor_query(
    anchor_text: str,
    anchor_relation: str,
    user_id: Any,
    pool: List[Dict[str, Any]],
    temporal_index,
    worker: Any
) -> List[int]:
    """
    Convenience function to resolve an anchor query and return matching block IDs.

    Args:
        anchor_text: The anchor event description
        anchor_relation: Temporal relation ("DURING", "BEFORE", "AFTER")
        user_id: User identifier
        pool: Pool of candidate blocks
        temporal_index: TemporalIndex instance
        worker: LLMWorker instance

    Returns:
        List of block IDs matching the resolved anchor query
    """
    resolver = AnchorResolver(temporal_index, worker)

    # Resolve anchor to a (start, end, time_type) triple.
    ref_start, ref_end, time_type = resolver.resolve_anchor(
        anchor_text, anchor_relation, user_id, pool
    )

    if time_type == "NONE" or not ref_start:
        # Fallback: return all blocks
        mx = _mx()
        return [mx._get_block_id(b) for b in pool]

    start_str = ref_start.isoformat()
    end_str = ref_end.isoformat() if ref_end else None

    matching_ids = temporal_index.query_temporal(
        start_date=start_str,
        end_date=end_str,
        time_type=time_type,
        use_event_time=True
    )

    return matching_ids
