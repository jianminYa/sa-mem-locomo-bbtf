"""
Interval Tree Index for efficient temporal range queries on memory blocks.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional


def _mx():
    """Lazy import to avoid circular imports."""
    import memblock_extractor as mx
    return mx


class IntervalNode:
    """Node in the interval tree."""
    def __init__(self, start: date, end: date, item_id: int, data: Dict[str, Any]):
        self.start = start
        self.end = end
        self.item_id = item_id
        self.data = data
        self.max_end = end


class IntervalTree:
    """Simple interval tree for temporal range queries."""

    def __init__(self):
        self.intervals: List[IntervalNode] = []
        self._sorted = False

    def insert(self, start: date, end: date, item_id: int, data: Dict[str, Any]):
        """Insert a time interval associated with one indexed item."""
        node = IntervalNode(start, end, item_id, data)
        self.intervals.append(node)
        self._sorted = False

    def _ensure_sorted(self):
        """Sort intervals by start time for efficient querying."""
        if not self._sorted:
            self.intervals.sort(key=lambda x: (x.start, x.end))
            self._sorted = True

    def query_range(self, query_start: date, query_end: date) -> List[int]:
        """
        Find all item_ids whose time intervals overlap with [query_start, query_end].
        Returns list of item_ids sorted by start time.
        """
        self._ensure_sorted()
        result: List[int] = []

        for node in self.intervals:
            # Overlap condition:
            # [a1, a2] and [b1, b2] overlap iff a1 <= b2 and b1 <= a2
            if node.start <= query_end and query_start <= node.end:
                result.append(node.item_id)

        return result

    def query_point(self, query_time: date) -> List[int]:
        """Find all item_ids whose time intervals contain query_time."""
        return self.query_range(query_time, query_time)

    def query_before(self, query_time: date) -> List[int]:
        """Find all item_ids that end before or at query_time."""
        self._ensure_sorted()
        result: List[int] = []

        for node in self.intervals:
            if node.end <= query_time:
                result.append(node.item_id)

        return result

    def query_after(self, query_time: date) -> List[int]:
        """Find all item_ids that start after or at query_time."""
        self._ensure_sorted()
        result: List[int] = []

        for node in self.intervals:
            if node.start >= query_time:
                result.append(node.item_id)

        return result


class TemporalIndex:
    """
    Temporal index for memory blocks using interval trees.
    Supports multiple temporal dimensions: session time and event time.
    """

    def __init__(self):
        # Separate trees for session time and event time.
        self.session_tree = IntervalTree()
        self.event_tree = IntervalTree()

        # Keyed by global_id (unique over all blocks passed to build_from_blocks).
        self.block_metadata: Dict[int, Dict[str, Any]] = {}

    def build_from_blocks(self, blocks: List[Dict[str, Any]]):
        """
        Build temporal index from memory blocks.

        IMPORTANT:
        - Uses global_id = enumerate index in `blocks` as the tree key.
        - Keeps original `block_id` in metadata for downstream display/compat.
        """
        mx = _mx()

        for global_id, block in enumerate(blocks):
            block_id = mx._get_block_id(block)

            # Extract temporal metadata.
            temporal_idx = block.get("temporal_index", {})

            # Session time range.
            session_start_str = temporal_idx.get("sessionstart_time", "Unknown")
            session_end_str = temporal_idx.get("sessionend_time", "Unknown")

            # Event time range.
            event_start_str = temporal_idx.get("block_event_start_time", "Unknown")
            event_end_str = temporal_idx.get("block_event_end_time", "Unknown")

            # Parse dates.
            session_start = self._parse_date(session_start_str)
            session_end = self._parse_date(session_end_str)
            event_start = self._parse_date(event_start_str)
            event_end = self._parse_date(event_end_str)

            # Store metadata under global_id.
            metadata = {
                "global_id": global_id,
                "block_id": block_id,
                "user_id": block.get("user_id"),
                "categories": block.get("features", {}).get("categories", []),
                "topic_kw_text": block.get("features", {}).get("topic_kw_text", ""),
                "events_count": block.get("events_count", 0),
                "temporal_index": temporal_idx,
            }
            self.block_metadata[global_id] = metadata

            # Insert into session tree using global_id.
            if session_start and session_end:
                self.session_tree.insert(session_start, session_end, global_id, metadata)

            # Insert into event tree using global_id.
            if event_start and event_end:
                self.event_tree.insert(event_start, event_end, global_id, metadata)

    @staticmethod
    def _parse_date(date_str: str) -> Optional[date]:
        """Parse ISO date string to date object."""
        if not date_str or date_str == "Unknown":
            return None

        try:
            if "T" in date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                return dt.date()
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return None

    def query_temporal(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        time_type: str = "RANGE",
        use_event_time: bool = True
    ) -> List[int]:
        """
        Query blocks by temporal constraints.

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            time_type: "POINT", "RANGE", "AFTER", "BEFORE", or "NONE"
            use_event_time: If True, use event time; otherwise use session time

        Returns:
            List of global_ids matching the temporal constraint.
        """
        if time_type == "NONE" or not start_date:
            return list(self.block_metadata.keys())

        tree = self.event_tree if use_event_time else self.session_tree

        start = self._parse_date(start_date)
        end = self._parse_date(end_date) if end_date else None

        if not start:
            return list(self.block_metadata.keys())

        t = (time_type or "RANGE").upper()

        if t in ("AFTER", "SINCE"):
            return tree.query_after(start)
        if t in ("BEFORE", "UNTIL"):
            return tree.query_before(start)
        if t == "POINT":
            return tree.query_point(start)
        if t == "RANGE":
            if end is None:
                return tree.query_range(start, date.max)
            return tree.query_range(start, end)

        # Fallback: no constraint.
        return list(self.block_metadata.keys())

    def query_by_categories(self, target_categories: List[str]) -> List[int]:
        """
        Query indexed items that contain any of target categories.

        Returns:
            List of global_ids.
        """
        if not target_categories:
            return list(self.block_metadata.keys())

        result: List[int] = []
        target_set = {cat.lower() for cat in target_categories}

        for global_id, metadata in self.block_metadata.items():
            block_categories = metadata.get("categories", [])
            block_cat_set = {cat.lower() for cat in block_categories}
            if block_cat_set & target_set:
                result.append(global_id)

        return result

    def query_by_topic_keywords(self, query_text: str, threshold: float = 0.3) -> List[int]:
        """
        Query indexed items by topic/keyword text overlap (token-based).

        Returns:
            List of global_ids with sufficient overlap.
        """
        if not query_text:
            return list(self.block_metadata.keys())

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return list(self.block_metadata.keys())

        result: List[int] = []

        for global_id, metadata in self.block_metadata.items():
            topic_kw = metadata.get("topic_kw_text", "")
            block_tokens = self._tokenize(topic_kw)

            if not block_tokens:
                continue

            overlap = len(query_tokens & block_tokens)
            ratio = overlap / len(query_tokens)

            if ratio >= threshold:
                result.append(global_id)

        return result

    @staticmethod
    def _tokenize(text: str) -> set:
        """Simple tokenization for keyword matching."""
        import re
        tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        stop_words = {
            "the", "a", "an", "of", "to", "in", "on", "for",
            "and", "or", "is", "are", "was", "were",
        }
        return {t for t in tokens if t and t not in stop_words}