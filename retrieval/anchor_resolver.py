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
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date
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

    def resolve_anchor(
        self,
        anchor_text: str,
        anchor_relation: str,
        user_id: Any,
        pool: List[Dict[str, Any]]
    ) -> Tuple[Optional[date], Optional[date]]:
        """
        Resolve an anchor event to a time range.

        Args:
            anchor_text: The anchor event description (e.g., "when I lived in Paris")
            anchor_relation: Temporal relation ("DURING", "BEFORE", "AFTER")
            user_id: User identifier
            pool: Pool of candidate blocks

        Returns:
            Tuple of (start_date, end_date) for the resolved anchor, or (None, None) if not found
        """
        mx = _mx()
        import time

        # Step 1: Find blocks that match the anchor event using vector similarity
        t_find_start = time.perf_counter()
        anchor_blocks = self._find_anchor_blocks(anchor_text, user_id, pool)
        t_find = time.perf_counter() - t_find_start

        if not anchor_blocks:
            mx.logger.warning("⚠️ No blocks found for anchor: '%s'", anchor_text)
            return None, None

        # Step 2: Extract time range from anchor blocks
        t_extract_start = time.perf_counter()
        anchor_start, anchor_end = self._extract_time_range(anchor_blocks)
        t_extract = time.perf_counter() - t_extract_start

        if not anchor_start or not anchor_end:
            mx.logger.warning("⚠️ Could not extract time range from anchor blocks")
            return None, None

        # Step 3: Adjust time range based on relation
        query_start, query_end = self._apply_relation(
            anchor_start, anchor_end, anchor_relation
        )

        mx.logger.info(
            "✅ Anchor resolved: '%s' %s -> [%s, %s]",
            anchor_text, anchor_relation, query_start, query_end
        )
        mx.logger.info(
            "⏱️ Anchor resolution timing: find_blocks=%.3fs, extract_time=%.3fs, total=%.3fs",
            t_find, t_extract, t_find + t_extract
        )

        return query_start, query_end

    def _find_anchor_blocks(
        self,
        anchor_text: str,
        user_id: Any,
        pool: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find blocks that best match the anchor event description.

        Uses vector similarity to find relevant blocks.
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

        # 优化：预先批量获取所有块的 embeddings（如果已缓存）
        # 而不是在循环中逐个获取
        scores = []
        for block in pool:
            block_id = mx._get_block_id(block)
            key = f"{user_id}_{block_id}"

            # Build block text
            features = block.get("features", {})
            topic_kw = features.get("topic_kw_text", "")

            events = block.get("events", [])
            event_texts = [e.get("description", "") for e in events if e.get("description")]
            event_str = " | ".join(event_texts)

            text = f"{topic_kw} {event_str}".strip()

            # 优化：使用已缓存的 embedding，避免重复计算
            block_vec = store.get_vector(
                key,
                "content_event_topic_kw",
                text,
                note=f"U{user_id}_B{block_id}_anchor",
            )

            try:
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity([anchor_vec], [block_vec])[0][0] if block_vec else -1.0
            except Exception:
                sim = -1.0

            scores.append((block, float(sim)))

        # Sort by similarity and take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_blocks = [block for block, score in scores[:top_k] if score > 0.3]

        store.flush()

        mx.logger.info(
            "🔍 Anchor resolution: found %d/%d blocks matching '%s' (processed %d blocks)",
            len(top_blocks), top_k, anchor_text[:30], len(pool)
        )

        return top_blocks

    def _extract_time_range(
        self,
        blocks: List[Dict[str, Any]]
    ) -> Tuple[Optional[date], Optional[date]]:
        """
        Extract the time range covered by a set of blocks.

        Returns the earliest start time and latest end time across all blocks.
        """
        start_dates = []
        end_dates = []

        for block in blocks:
            temporal_idx = block.get("temporal_index", {})

            # Try event time first (more specific)
            event_start_str = temporal_idx.get("block_event_start_time")
            event_end_str = temporal_idx.get("block_event_end_time")

            event_start = self._parse_date(event_start_str)
            event_end = self._parse_date(event_end_str)

            if event_start:
                start_dates.append(event_start)
            if event_end:
                end_dates.append(event_end)

            # Fallback to session time if event time not available
            if not event_start or not event_end:
                session_start_str = temporal_idx.get("sessionstart_time")
                session_end_str = temporal_idx.get("sessionend_time")

                session_start = self._parse_date(session_start_str)
                session_end = self._parse_date(session_end_str)

                if session_start and not event_start:
                    start_dates.append(session_start)
                if session_end and not event_end:
                    end_dates.append(session_end)

        if not start_dates or not end_dates:
            return None, None

        # Return the full range covered by all blocks
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
    ) -> Tuple[date, date]:
        """
        Apply temporal relation to anchor time range.

        Args:
            anchor_start: Start of anchor period
            anchor_end: End of anchor period
            relation: "DURING", "BEFORE", or "AFTER"

        Returns:
            Query time range based on relation
        """
        from datetime import timedelta

        if relation == "DURING":
            # Query within the anchor period
            return anchor_start, anchor_end

        elif relation == "BEFORE":
            # Query before the anchor period
            # Use a reasonable lookback window (e.g., 10 years before)
            lookback_start = anchor_start - timedelta(days=365 * 10)
            return lookback_start, anchor_start

        elif relation == "AFTER":
            # Query after the anchor period
            # Use a reasonable lookahead window (e.g., 10 years after)
            lookahead_end = anchor_end + timedelta(days=365 * 10)
            return anchor_end, lookahead_end

        else:
            # Default: use anchor period
            return anchor_start, anchor_end


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

    # Resolve anchor to time range
    start_date, end_date = resolver.resolve_anchor(
        anchor_text, anchor_relation, user_id, pool
    )

    if not start_date or not end_date:
        # Fallback: return all blocks
        mx = _mx()
        return [mx._get_block_id(b) for b in pool]

    # Query temporal index with resolved time range
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    matching_ids = temporal_index.query_temporal(
        start_date=start_str,
        end_date=end_str,
        time_type="RANGE",
        use_event_time=True
    )

    return matching_ids
