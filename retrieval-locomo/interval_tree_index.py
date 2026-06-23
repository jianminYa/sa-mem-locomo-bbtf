"""
Interval Tree Index for efficient temporal range queries on memory blocks.

Implements a real centered interval tree:
- Build: O(n log n)
- query_range / query_point: O(log n + k)
- query_before / query_after: O(log n + k) via sorted endpoint arrays

Public surface (IntervalNode, IntervalTree.insert/query_range/query_point/
query_before/query_after) is preserved for backward compatibility.
"""
from __future__ import annotations

import bisect
from datetime import datetime, date
from typing import Any, Dict, List, Optional


def _mx():
    """Lazy import to avoid circular imports."""
    import memblock_extractor as mx
    return mx


class IntervalNode:
    """Node in the interval tree (one inserted interval)."""
    __slots__ = ("start", "end", "item_id", "data", "max_end")

    def __init__(self, start: date, end: date, item_id: int, data: Dict[str, Any]):
        self.start = start
        self.end = end
        self.item_id = item_id
        self.data = data
        self.max_end = end


class _CenterNode:
    """Internal node of the centered interval tree."""
    __slots__ = ("center", "left", "right", "by_start_asc", "by_end_desc")

    def __init__(self, center: date):
        self.center: date = center
        self.left: Optional["_CenterNode"] = None
        self.right: Optional["_CenterNode"] = None
        # Intervals containing `center`, ordered two ways for efficient partial scans.
        self.by_start_asc: List[IntervalNode] = []
        self.by_end_desc: List[IntervalNode] = []


class IntervalTree:
    """Centered interval tree for temporal range queries."""

    def __init__(self):
        self.intervals: List[IntervalNode] = []
        self._sorted = False  # kept for backward compatibility
        self._root: Optional[_CenterNode] = None
        # Auxiliary sorted arrays used by query_before / query_after.
        self._by_start: List[IntervalNode] = []
        self._by_end: List[IntervalNode] = []
        self._start_keys: List[date] = []
        self._end_keys: List[date] = []
        self._dirty = True

    def insert(self, start: date, end: date, item_id: int, data: Dict[str, Any]):
        """Insert a time interval associated with one indexed item."""
        # Guard against inverted intervals (start > end) which cause infinite
        # loops in _build_tree because the item can never land in 'middle'.
        if start > end:
            start, end = end, start
        node = IntervalNode(start, end, item_id, data)
        self.intervals.append(node)
        self._dirty = True
        self._sorted = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _ensure_built(self):
        """Lazily (re)build the tree after inserts."""
        if not self._dirty:
            return
        self._root = self._build_tree(list(self.intervals))
        self._by_start = sorted(self.intervals, key=lambda n: (n.start, n.end))
        self._by_end = sorted(self.intervals, key=lambda n: (n.end, n.start))
        self._start_keys = [n.start for n in self._by_start]
        self._end_keys = [n.end for n in self._by_end]
        self._dirty = False
        self._sorted = True

    # Backward-compat shim (used to be internal).
    def _ensure_sorted(self):
        self._ensure_built()

    @staticmethod
    def _build_tree(items: List[IntervalNode]) -> Optional[_CenterNode]:
        """Iterative centered interval tree build (avoids Python RecursionError)."""
        if not items:
            return None

        root_holder: List[Optional[_CenterNode]] = [None]
        # Stack entries: (items_list, parent_node_or_None, side: "left"/"right"/None)
        stack: List[tuple] = [(items, None, None)]

        while stack:
            cur_items, parent, side = stack.pop()

            if not cur_items:
                if parent is not None:
                    setattr(parent, side, None)
                continue

            # Median of all endpoints -> balances left/right partitions.
            endpoints: List[date] = []
            for n in cur_items:
                endpoints.append(n.start)
                endpoints.append(n.end)
            endpoints.sort()
            center = endpoints[len(endpoints) // 2]

            node = _CenterNode(center)

            left_items: List[IntervalNode] = []
            right_items: List[IntervalNode] = []
            middle: List[IntervalNode] = []

            for n in cur_items:
                if n.end < center:
                    left_items.append(n)
                elif n.start > center:
                    right_items.append(n)
                else:
                    middle.append(n)

            node.by_start_asc = sorted(middle, key=lambda n: (n.start, n.end))
            node.by_end_desc = sorted(middle, key=lambda n: (n.end, n.start), reverse=True)

            if parent is None:
                root_holder[0] = node
            else:
                setattr(parent, side, node)

            # Degenerate: all items fall into middle (center didn't split anything).
            if not left_items and not right_items:
                node.left = None
                node.right = None
            else:
                stack.append((left_items, node, "left"))
                stack.append((right_items, node, "right"))

        return root_holder[0]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_range(self, query_start: date, query_end: date) -> List[int]:
        """
        Find all item_ids whose intervals overlap [query_start, query_end].
        Result is stable-sorted by (start, end) to match the legacy ordering.
        """
        self._ensure_built()
        collected: List[IntervalNode] = []
        self._query_range(self._root, query_start, query_end, collected)
        collected.sort(key=lambda n: (n.start, n.end))
        return [n.item_id for n in collected]

    @staticmethod
    def _query_range(
        node: Optional[_CenterNode],
        qs: date,
        qe: date,
        out: List[IntervalNode],
    ) -> None:
        """Iterative range query (avoids RecursionError on deep trees)."""
        stack: List[Optional[_CenterNode]] = [node]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            c = node.center

            if qe < c:
                # Query range lies strictly left of center: only middle intervals
                # with start <= qe can overlap.
                for iv in node.by_start_asc:
                    if iv.start > qe:
                        break
                    out.append(iv)
                stack.append(node.left)
            elif qs > c:
                # Query range lies strictly right of center: only middle intervals
                # with end >= qs can overlap.
                for iv in node.by_end_desc:
                    if iv.end < qs:
                        break
                    out.append(iv)
                stack.append(node.right)
            else:
                # Query range straddles center -> all middle intervals overlap.
                out.extend(node.by_start_asc)
                stack.append(node.left)
                stack.append(node.right)

    def query_point(self, query_time: date) -> List[int]:
        """Find all item_ids whose intervals contain query_time."""
        return self.query_range(query_time, query_time)

    def query_before(self, query_time: date) -> List[int]:
        """Find all item_ids whose intervals end at or before query_time."""
        self._ensure_built()
        if not self._end_keys:
            return []
        # bisect_right on end_keys gives count of intervals with end <= query_time.
        idx = bisect.bisect_right(self._end_keys, query_time)
        hits = self._by_end[:idx]
        hits.sort(key=lambda n: (n.start, n.end))
        return [n.item_id for n in hits]

    def query_after(self, query_time: date) -> List[int]:
        """Find all item_ids whose intervals start at or after query_time."""
        self._ensure_built()
        if not self._start_keys:
            return []
        # bisect_left gives index of first interval with start >= query_time.
        idx = bisect.bisect_left(self._start_keys, query_time)
        hits = self._by_start[idx:]
        hits.sort(key=lambda n: (n.start, n.end))
        return [n.item_id for n in hits]


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