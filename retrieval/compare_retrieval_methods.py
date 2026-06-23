"""
Enhanced retrieval with 5-mode ablation study support.
Based on retrieval_enhanced_old.py (high accuracy version).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import math
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from query_pasing_byllm import QueryParser, SearchDirective
from interval_tree_index import TemporalIndex
from datetime import datetime, date

def _mx():
    """Lazy import to avoid circular imports."""
    import memblock_extractor as mx
    return mx


class EnhancedRetriever:
    """
    Enhanced retrieval system with granular ablation study support:
    - baseline (B0): No parsing, no filtering
    - time_only (T): Parse + temporal filter only
    - time_event (T): Same as time_only (event type filter disabled)
    - full (T): Same as time_only (anchor + event type filters disabled)

    Legacy modes (for backward compatibility):
    - parse_only: Parse but no filtering
    - event_only: Same as parse_only (event type filter disabled)

    Note: Event type filtering has been disabled due to performance degradation.
    """

    def __init__(self, worker: Any, top_k: int = None):
        mx = _mx()
        self.worker = worker
        self.top_k = mx.Config.TOP_K_RETRIEVE if top_k is None else top_k

        # Query parser for intent and time extraction
        self.query_parser = QueryParser(
            api_key=mx.Config.API_KEY,
            base_url=mx.Config.BASE_URL,
            model=mx.Config.LLM_MODEL
        )

        # Data structures
        self.all_blocks: List[Dict[str, Any]] = []
        self.temporal_index: TemporalIndex = TemporalIndex()
        self.blocks_by_user: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        self.trace_map: Dict[Any, Dict[int, List[int]]] = {}

        # Ground truth for ablation study
        self.ground_truth: Dict[str, List[int]] = {}

    def load(self):
        """Load memory blocks and build indices."""
        mx = _mx()

        # Clear existing data to avoid accumulation
        self.all_blocks = []
        self.blocks_by_user = defaultdict(list)

        # Load blocks
        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            self.all_blocks = [json.loads(l) for l in f if l.strip()]

        # Group by user
        for block in self.all_blocks:
            user_id = block.get("user_id")
            if user_id is not None:
                self.blocks_by_user[user_id].append(block)

        # Build temporal index
        self.temporal_index.build_from_blocks(self.all_blocks)

        # Load traces (for compatibility)
        self.trace_map = self._load_traces(mx.Config.TIME_TRACE_FILE)

        # Load ground truth for ablation study
        self._load_ground_truth()

        mx.logger.info("✅ Loaded %d blocks, built temporal index", len(self.all_blocks))

    @staticmethod
    def _load_traces(path: str) -> Dict[Any, Dict[int, List[int]]]:
        """Load trace mapping from file."""
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

    @staticmethod
    def _iso_to_date(s: str) -> Optional[date]:
        if not s:
            return None
        try:
            # supports "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except Exception:
            try:
                return datetime.strptime(s.split("T")[0], "%Y-%m-%d").date()
            except Exception:
                return None
    @staticmethod
    def _time_decay_score(block_day: Optional[date], query_day: Optional[date], intent: str) -> float:
        """
        Soft time scoring:
        - Decay with distance in days
        - For AS_OF/STATIC: penalize evidence that is AFTER the query_day (future leakage)
        """
        if not block_day or not query_day:
            return 1.0

        delta = (block_day - query_day).days
        # half-life-ish: tau controls how fast relevance drops with time distance
        # tau=1500 matches buffer_days, so blocks at edge of hard filter get ~0.37 score
        tau = 1500.0
        base = math.exp(-abs(delta) / tau)

        intent = (intent or "").upper()
        if intent in ("AS_OF", "STATIC") and delta > 0:
            # evidence timestamp later than "as of" date is usually harmful
            base *= 0.3

        return float(base)           
    
    def _load_ground_truth(self):
        """Load ground truth for ablation study."""
        mx = _mx()
        gt_path = os.path.join(mx.Config.OUTPUT_DIR, "ground_truth.json")

        if not os.path.exists(gt_path):
            mx.logger.warning("⚠️ Ground truth file not found: %s", gt_path)
            return

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            # Extract ground_truth dict from the nested structure
            if "ground_truth" in gt_data:
                gt_dict = gt_data["ground_truth"]
            else:
                gt_dict = gt_data

            # Build lookup: (user_id, qa_idx) -> [block_ids]
            # Since we only have one user per file, we need to get user_id from blocks
            # For now, we'll store by qa_idx only and match with user_id later
            for qa_idx_str, entry in gt_dict.items():
                if isinstance(entry, dict):
                    block_ids = entry.get("block_ids", [])
                    # Store by qa_idx as string for now
                    self.ground_truth[qa_idx_str] = block_ids

            mx.logger.info("✅ Loaded ground truth for %d queries", len(self.ground_truth))
        except Exception as e:
            mx.logger.warning("⚠️ Failed to load ground truth: %s", e)

    def _filter_by_metadata(
        self,
        pool: List[Dict[str, Any]],
        directive: SearchDirective,
        mode: str,
        gt_blocks: Optional[List[int]] = None,
        category: Optional[str] = None,
        enable_event_type: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter memory blocks based on parsed query metadata.

        Args:
            pool: Initial pool of candidate blocks
            directive: Parsed query directive with time and topic constraints
            mode: Filtering mode - "time_only", "event_only", "full", etc.
            gt_blocks: Ground truth block IDs for tracking
            category: Query category (e.g., "Memory Conflict")
            enable_event_type: Whether to enable event type filtering

        Returns:
            Tuple of (filtered blocks, filtering_trace)
        """
        mx = _mx()

        # Extract block IDs from pool
        pool_ids = {mx._get_block_id(b) for b in pool}

        # Initialize filtering trace
        gt_set = set(gt_blocks) if gt_blocks else set()
        filtering_trace = {
            "pool_ids": list(pool_ids),
            "pool_size": len(pool_ids),
            "gt_in_pool": len(pool_ids & gt_set),
            "gt_total": len(gt_set),
        }

        # 1. Temporal filtering (for time_only,time_event, and full modes)
        time_filtered_ids = pool_ids
        if mode in ["time_only", "time_event", "full"]:
            if category == "Memory Conflict":
                mx.logger.info("⚠️ Skipping time filtering for Memory Conflict question")
                time_filtered_ids = pool_ids
            elif directive.use_interval_tree and directive.time_constraint.type != "NONE":
                tc = directive.time_constraint
                intent = directive.intent

                def _temporal_query_ids(
                    start_date: Optional[str],
                    end_date: Optional[str],
                    time_type: str,
                    use_event_time: bool,
                ) -> set:
                    ids = set(self.temporal_index.query_temporal(
                        start_date=start_date,
                        end_date=end_date,
                        time_type=time_type,
                        use_event_time=use_event_time,
                    ))
                    return ids & pool_ids

                # Anchor 已移除：如果 parser 仍产出 ANCHOR，这里直接忽略该约束
                if tc.type == "ANCHOR":
                    mx.logger.info("⚠️ ANCHOR time constraint ignored (anchor disabled)")
                    tc.type = "NONE"

                # Non-anchor temporal filtering
                if intent in ["PLANNING", "STATIC"] and tc.type in ["RANGE", "POINT", "AFTER", "BEFORE"]:
                    # For STATIC/PLANNING: Skip hard filtering, use soft time decay scoring
                    # These queries need historical evidence, not events within the time window
                    mx.logger.info(
                        "🔍 %s query: Skipping hard time filter (will use soft time decay scoring)",
                        intent
                    )
                    time_filtered_ids = pool_ids

                elif intent in ["WINDOW", "AS_OF"] and tc.type in ["RANGE", "POINT", "AFTER", "BEFORE"]:
                    from datetime import datetime, timedelta
                    start_str = tc.start
                    end_str = tc.end

                    if start_str:
                        try:
                            # Use different buffer_days for WINDOW vs AS_OF
                            if intent == "WINDOW":
                                buffer_days = 365  # WINDOW: 1 year buffer
                            else:  # AS_OF
                                buffer_days = 180  # AS_OF: 6 month buffer (tighter)

                            if tc.type == "AFTER":
                                start_date = datetime.fromisoformat(start_str.split('T')[0])
                                expanded_start = (start_date - timedelta(days=buffer_days)).isoformat()
                                ids_evt = _temporal_query_ids(expanded_start, None, "AFTER", True)
                                mx.logger.info(
                                    "🔍 %s temporal(event) filter: AFTER [%s, +inf) (buffered) -> %d/%d blocks",
                                    intent, expanded_start, len(ids_evt), len(pool_ids)
                                )

                            elif tc.type == "BEFORE":
                                # Use tc.end if present, else fallback to tc.start as cutoff
                                cutoff_str = (end_str or start_str)
                                cutoff_date = datetime.fromisoformat(cutoff_str.split('T')[0])
                                expanded_cutoff = (cutoff_date + timedelta(days=buffer_days)).isoformat()
                                ids_evt = _temporal_query_ids(expanded_cutoff, None, "BEFORE", True)
                                mx.logger.info(
                                    "🔍 %s temporal(event) filter: BEFORE (-inf, %s] (buffered) -> %d/%d blocks",
                                    intent, expanded_cutoff, len(ids_evt), len(pool_ids)
                                )

                            else:
                                start_date = datetime.fromisoformat(start_str.split('T')[0])
                                expanded_start = (start_date - timedelta(days=buffer_days)).isoformat()

                                if end_str:
                                    end_date = datetime.fromisoformat(end_str.split('T')[0])
                                    expanded_end = (end_date + timedelta(days=buffer_days)).isoformat()
                                else:
                                    expanded_end = (start_date + timedelta(days=buffer_days)).isoformat()

                                ids_evt = _temporal_query_ids(expanded_start, expanded_end, "RANGE", True)
                                mx.logger.info(
                                    "🔍 %s temporal(event) filter: RANGE [%s, %s] (buffered) -> %d/%d blocks",
                                    intent, expanded_start, expanded_end, len(ids_evt), len(pool_ids)
                                )

                            if len(ids_evt) == 0:
                                if tc.type == "AFTER":
                                    ids_sess = _temporal_query_ids(expanded_start, None, "AFTER", False)
                                elif tc.type == "BEFORE":
                                    ids_sess = _temporal_query_ids(expanded_cutoff, None, "BEFORE", False)
                                else:
                                    ids_sess = _temporal_query_ids(expanded_start, expanded_end, "RANGE", False)

                                mx.logger.info(
                                    "🔍 %s temporal(event) empty, trying session-time -> %d/%d blocks",
                                    intent, len(ids_sess), len(pool_ids)
                                )
                                time_filtered_ids = ids_sess
                            else:
                                time_filtered_ids = ids_evt
                        except Exception as e:
                            mx.logger.warning("⚠️ Failed to parse %s date: %s", intent, e)
                            ids_evt = _temporal_query_ids(tc.start, tc.end, tc.type, True)
                            time_filtered_ids = ids_evt if len(ids_evt) > 0 else _temporal_query_ids(tc.start, tc.end, tc.type, False)
                    else:
                        time_filtered_ids = pool_ids

                else:
                    from datetime import datetime, timedelta
                    start_str = tc.start
                    end_str = tc.end

                    if start_str and tc.type in ["RANGE", "POINT", "AFTER", "BEFORE"]:
                        try:
                            if tc.type == "AFTER":
                                start_date = datetime.fromisoformat(start_str.split('T')[0])
                                expanded_start = (start_date - timedelta(days=1500)).isoformat()
                                ids_evt = _temporal_query_ids(expanded_start, None, "AFTER", True)
                                mx.logger.info(
                                    "🔍 Temporal(event) filter: AFTER [%s, +inf) (expanded) -> %d/%d blocks",
                                    expanded_start, len(ids_evt), len(pool_ids)
                                )

                            elif tc.type == "BEFORE":
                                cutoff_str = (end_str or start_str)
                                cutoff_date = datetime.fromisoformat(cutoff_str.split('T')[0])
                                expanded_cutoff = (cutoff_date + timedelta(days=1500)).isoformat()
                                ids_evt = _temporal_query_ids(expanded_cutoff, None, "BEFORE", True)
                                mx.logger.info(
                                    "🔍 Temporal(event) filter: BEFORE (-inf, %s] (expanded) -> %d/%d blocks",
                                    expanded_cutoff, len(ids_evt), len(pool_ids)
                                )

                            else:
                                start_date = datetime.fromisoformat(start_str.split('T')[0])
                                expanded_start = (start_date - timedelta(days=1500)).isoformat()

                                if end_str:
                                    end_date = datetime.fromisoformat(end_str.split('T')[0])
                                    expanded_end = (end_date + timedelta(days=1500)).isoformat()
                                else:
                                    expanded_end = (start_date + timedelta(days=1500)).isoformat()

                                ids_evt = _temporal_query_ids(expanded_start, expanded_end, "RANGE", True)
                                mx.logger.info(
                                    "🔍 Temporal(event) filter: RANGE [%s, %s] (expanded) -> %d/%d blocks",
                                    expanded_start, expanded_end, len(ids_evt), len(pool_ids)
                                )

                            if len(ids_evt) == 0:
                                if tc.type == "AFTER":
                                    ids_sess = _temporal_query_ids(expanded_start, None, "AFTER", False)
                                elif tc.type == "BEFORE":
                                    ids_sess = _temporal_query_ids(expanded_cutoff, None, "BEFORE", False)
                                else:
                                    ids_sess = _temporal_query_ids(expanded_start, expanded_end, "RANGE", False)

                                mx.logger.info(
                                    "🔍 Temporal(event) empty, trying session-time -> %d/%d blocks",
                                    len(ids_sess), len(pool_ids)
                                )
                                time_filtered_ids = ids_sess
                            else:
                                time_filtered_ids = ids_evt
                        except Exception as e:
                            mx.logger.warning("⚠️ Failed to parse date for expansion: %s", e)
                            ids_evt = _temporal_query_ids(tc.start, tc.end, tc.type, True)
                            time_filtered_ids = ids_evt if len(ids_evt) > 0 else _temporal_query_ids(tc.start, tc.end, tc.type, False)
                    else:
                        ids_evt = _temporal_query_ids(tc.start, tc.end, tc.type, True)
                        mx.logger.info(
                            "🔍 Temporal(event) filter: %s [%s, %s] -> %d/%d blocks",
                            tc.type, tc.start, tc.end, len(ids_evt), len(pool_ids)
                        )
                        if len(ids_evt) == 0:
                            ids_sess = _temporal_query_ids(tc.start, tc.end, tc.type, False)
                            mx.logger.info(
                                "🔍 Temporal(event) empty, trying session-time -> %d/%d blocks",
                                len(ids_sess), len(pool_ids)
                            )
                            time_filtered_ids = ids_sess
                        else:
                            time_filtered_ids = ids_evt

        # Track GT metrics after temporal filtering
        gt_after_time = len(time_filtered_ids & gt_set)
        gt_lost_time = len(gt_set - time_filtered_ids)
        filtering_trace.update({
            "time_filtered_ids": list(time_filtered_ids),
            "time_filtered_size": len(time_filtered_ids),
            "gt_after_time": gt_after_time,
            "gt_lost_time": gt_lost_time,
            "after_time_filtering": {
                "block_ids": list(time_filtered_ids),
                "size": len(time_filtered_ids),
            }
        })

        # 2. Event type filtering (P0: Soft Gate + P1: Compatible Types)
        # Use enable_event_type parameter to control whether to apply event type filtering
        # For modes without time filtering (event_only), start from pool_ids
        # For modes with time filtering, start from time_filtered_ids
        if mode == "event_only":
            category_filtered_ids = pool_ids
        else:
            category_filtered_ids = time_filtered_ids

        # P1: Compatible type mapping
        # Maps target types to compatible types that should be accepted
        COMPATIBLE_TYPES = {
            "ATTRIBUTE": {"ATTRIBUTE", "STATE"},  # Preferences/interests often expressed as states
            "OCCURRENCE": {"OCCURRENCE", "STATE"},  # Events causing state changes
            "INTENTION": {"INTENTION", "STATE"},  # Plans/intentions often expressed as states
            "STATE": {"STATE"},  # State only matches state
        }

        # Compute event type match scores for each block
        event_type_scores = {}  # block_id -> score multiplier
        type_filtered_ids = category_filtered_ids  # Keep all blocks (no hard filtering)

        if enable_event_type:  # Use parameter instead of checking mode
            # Short-term fix: Disable event type filtering for PLANNING and MISC intents
            # These intents need diverse evidence types (STATE, OCCURRENCE, INTENTION)
            if directive.intent in ["PLANNING", "MISC"]:
                mx.logger.info(
                    "🔍 Event type scoring: DISABLED for intent=%s (needs diverse evidence)",
                    directive.intent
                )
                for block_id in category_filtered_ids:
                    event_type_scores[block_id] = 1.0
            elif directive.target_types and len(directive.target_types) < 4:
                # Only apply scoring if specific types are targeted (not all types)
                target_set = set(directive.target_types)

                # Expand target set with compatible types
                expanded_target_set = set()
                for target_type in target_set:
                    expanded_target_set.update(COMPATIBLE_TYPES.get(target_type, {target_type}))

                exact_matches = set()
                compatible_matches = set()
                no_matches = set()

                for block in pool:
                    block_id = mx._get_block_id(block)
                    if block_id not in category_filtered_ids:
                        continue

                    # Check if block has events of target types
                    events = block.get("events", [])
                    block_event_types = set()
                    for event in events:
                        event_type = event.get("event_temporal_type", "ATTRIBUTE")
                        block_event_types.add(event_type)

                    # Classify match quality
                    if block_event_types & target_set:
                        # Exact match: block has at least one event of exact target type
                        exact_matches.add(block_id)
                        event_type_scores[block_id] = 1.0
                    elif block_event_types & expanded_target_set:
                        # Compatible match: block has compatible type but not exact
                        compatible_matches.add(block_id)
                        event_type_scores[block_id] = 0.85
                    else:
                        # No match: block has no relevant types
                        no_matches.add(block_id)
                        event_type_scores[block_id] = 0.7

                mx.logger.info(
                    "🔍 Event type scoring: %s -> exact=%d, compatible=%d, other=%d (total=%d)",
                    directive.target_types, len(exact_matches), len(compatible_matches),
                    len(no_matches), len(category_filtered_ids)
                )
            else:
                # All types targeted, no scoring needed
                for block_id in category_filtered_ids:
                    event_type_scores[block_id] = 1.0
        else:
            # Event type filtering disabled, all blocks get score 1.0
            for block_id in category_filtered_ids:
                event_type_scores[block_id] = 1.0

        # Track GT metrics after event type filtering (no blocks removed, so same as before)
        gt_after_event = len(type_filtered_ids & gt_set)
        gt_lost_event = 0  # No blocks removed with soft gate
        filtering_trace.update({
            "event_filtered_ids": list(type_filtered_ids),
            "event_filtered_size": len(type_filtered_ids),
            "gt_after_event": gt_after_event,
            "gt_lost_event": gt_lost_event,
            "final_ids": list(type_filtered_ids),
            "final_size": len(type_filtered_ids),
            "event_type_scores": event_type_scores,  # Store scores for ranking
        })

        # Return filtered pool (all blocks from category filter, no hard removal)
        filtered_pool = [b for b in pool if mx._get_block_id(b) in type_filtered_ids]

        return filtered_pool, filtering_trace

    def _score_and_rank(
        self,
        user_id: Any,
        qa: Dict[str, Any],
        mode: str = "full",
        force_llm: bool = False
    ) -> Tuple[Dict[str, List[int]], Dict[int, float], List[int], Dict[str, Any]]:
        """
        Score and rank blocks for a query with optional metadata filtering.

        Args:
            user_id: User identifier
            qa: Question-answer pair
            mode: Filtering mode - "baseline", "parse_only", "time_only", "event_only", "full"
            force_llm: If True, force LLM parsing (disable fast path)

        Returns:
            Tuple of (rankings, similarity_map, target_boxes, filtering_trace)
        """
        mx = _mx()

        pool = self.blocks_by_user.get(user_id, [])
        if not pool:
            return {}, {}, [], {}

        question = qa.get("question", "") or ""
        q_id = qa.get("id", qa.get("question", ""))
        qa_idx = qa.get("qa_idx", q_id)

        # Get ground truth for this query (using qa_idx as string)
        gt_key = str(qa_idx)
        gt_blocks = self.ground_truth.get(gt_key, [])

        # Initialize filtering trace
        filtering_trace = {
            "mode": mode,
            "user_id": user_id,
            "qa_idx": qa_idx,
            "question": question,
            "force_llm": force_llm,
        }

        # Parse query (skip for baseline mode)
        directive = None
        if mode != "baseline" and question:
            try:
                # Force LLM parsing if requested
                if force_llm:
                    # Temporarily disable fast path
                    original_fast_path = self.query_parser.use_fast_path
                    self.query_parser.use_fast_path = False
                    directive = self.query_parser.parse(question)
                    self.query_parser.use_fast_path = original_fast_path
                else:
                    directive = self.query_parser.parse(question)

                mx.logger.info(
                    "📝 Query parsed: intent=%s, time=%s, source=%s",
                    directive.intent, directive.time_constraint.type, directive.parse_source
                )
                filtering_trace["parse_source"] = directive.parse_source
                filtering_trace["intent"] = directive.intent
                filtering_trace["time_constraint_type"] = directive.time_constraint.type
                # Save full directive to trace for analysis
                filtering_trace["directive"] = {
                    "intent": directive.intent,
                    "time_constraint": {
                        "type": directive.time_constraint.type,
                        "start": directive.time_constraint.start,
                        "end": directive.time_constraint.end,
                    },
                    "target_types": directive.target_types,
                    "rewritten_query": directive.rewritten_query,
                    "parse_source": directive.parse_source,
                }
            except Exception as e:
                mx.logger.warning("⚠️ Query parsing failed: %s", e)
                directive = None

        # Apply metadata filtering based on mode
        filtered_pool = pool
        filter_trace = {}

        # Define mode configurations
        mode_config = {
            "baseline": {"parse": False, "time": False, "event": False},
            "parse_only": {"parse": True, "time": False, "event": False},
            "time_only": {"parse": True, "time": True, "event": False},   # T
            "time_event": {"parse": True, "time": True, "event": False},  # T+E (event disabled)
            "full": {"parse": True, "time": True, "event": False},        # T only (anchor + event disabled)
            "event_only": {"parse": True, "time": False, "event": False}, # event disabled
        }

        config = mode_config.get(mode, mode_config["full"])

        if (config["time"] or config["event"]) and directive:
            category = qa.get("category", "")
            filtered_pool, filter_trace = self._filter_by_metadata(
                pool, directive, mode, gt_blocks, category,
                enable_event_type=config["event"]
            )

            # Fallback: If filtering is too aggressive (< 3 blocks), use full pool
            if len(filtered_pool) < 3:
                mx.logger.warning(
                    "⚠️ Metadata filtering too aggressive (%d blocks), using full pool",
                    len(filtered_pool)
                )
                filtered_pool = pool
                filter_trace["fallback_triggered"] = True
        else:
            # For baseline and parse_only modes, create minimal trace
            pool_ids = [mx._get_block_id(b) for b in pool]
            gt_set = set(gt_blocks)
            filter_trace = {
                "pool_ids": pool_ids,
                "pool_size": len(pool_ids),
                "gt_in_pool": len(set(pool_ids) & gt_set),
                "gt_total": len(gt_set),
                "time_filtered_ids": pool_ids,
                "time_filtered_size": len(pool_ids),
                "gt_after_time": len(set(pool_ids) & gt_set),
                "gt_lost_time": 0,
                "event_filtered_ids": pool_ids,
                "event_filtered_size": len(pool_ids),
                "gt_after_event": len(set(pool_ids) & gt_set),
                "gt_lost_event": 0,
                "final_ids": pool_ids,
                "final_size": len(pool_ids),
                "fallback_triggered": False,
            }

        # Merge filter trace into filtering_trace
        filtering_trace.update(filter_trace)

        # Vector similarity ranking on filtered pool
        store = mx.EmbeddingStore(self.worker, user_id)

        # Use rewritten query if available, otherwise use original
        query_text = directive.rewritten_query if directive else question
        qvec = store.get_vector(
            f"qa_{user_id}_{q_id}",
            "question",
            query_text,
            note=f"U{user_id}_QA_Enhanced"
        )
        # prepare query_day for time decay
        query_day = None
        if directive and directive.time_constraint and directive.time_constraint.start:
            query_day = self._iso_to_date(directive.time_constraint.start)

        sim_map: Dict[int, float] = {}
        for b in filtered_pool:
            bid = mx._get_block_id(b)
            key = f"{user_id}_{bid}"

            # Build enriched text for embedding
            features = b.get("features", {})
            topic_kw = features.get("topic_kw_text", "")

            # Include event descriptions
            events = b.get("events", [])
            event_texts = [e.get("description", "") for e in events if e.get("description")]
            event_str = " | ".join(event_texts[:5])  # Limit to first 5 events

            text = f"{topic_kw} {event_str}".strip()

            v = store.get_vector(
                key,
                "content_event_topic_kw",
                text,
                note=f"U{user_id}_B{bid}_enhanced",
            )

            try:
                s = cosine_similarity([qvec], [v])[0][0] if v else -1.0
            except Exception:
                s = -1.0

            # Apply event type score multiplier (P0: Soft Gate)
            event_type_score = filtering_trace.get("event_type_scores", {}).get(bid, 1.0)

            # ✅ soft time decay scoring
            time_score = 1.0
            if directive and query_day:
                # Determine if time decay should be applied based on mode and intent
                should_apply_decay = False

                if mode == "time_only":
                    # TIME_ONLY: Apply decay to all queries with time constraints
                    should_apply_decay = (directive.time_constraint and
                                         directive.time_constraint.type != "NONE")
                else:
                    # Other modes: Only STATIC/PLANNING (AS_OF now uses hard filtering)
                    should_apply_decay = directive.intent in ("STATIC", "PLANNING")

                if should_apply_decay:
                    tdx = (b.get("temporal_index") or {})
                    # use session end as "observed time" proxy
                    block_day = self._iso_to_date(tdx.get("sessionend_time") or
                                                  tdx.get("sessionstart_time") or "")
                    time_score = self._time_decay_score(block_day, query_day, directive.intent)

            final_score = float(s) * float(time_score)
            sim_map[bid] = final_score


        # Rank by similarity (with event type scoring applied)
        ranked = [bid for bid, _ in sorted(sim_map.items(), key=lambda x: x[1], reverse=True)]

        rankings = {"content_event_topic_kw": ranked}

        store.flush()

        # Track GT metrics in final ranked results
        gt_set = set(gt_blocks)
        ranked_set = set(ranked[:self.top_k])
        filtering_trace.update({
            "final_ranked_ids": ranked[:self.top_k],
            "gt_in_topk": len(ranked_set & gt_set),
        })

        # Get target boxes for evaluation
        target_boxes = mx.evidence_to_targets(qa.get("evidence"), pool)

        return rankings, sim_map, target_boxes, filtering_trace

    def run(
        self,
        result_jsonl: str,
        result_csv: str,
        mode: str = "full",
        force_llm: bool = False
    ):
        """
        Run retrieval on all queries.

        Args:
            result_jsonl: Output JSONL file path
            result_csv: Output CSV file path
            mode: Filtering mode - "baseline", "parse_only", "time_only", "event_only", "full"
            force_llm: If True, force LLM parsing (disable fast path)
        """
        mx = _mx()

        # Check if blocks are already built
        if not os.path.exists(mx.Config.FINAL_CONTENT_FILE):
            if not os.path.exists(mx.Config.RAW_DATA_FILE):
                mx.logger.error("❌ No raw data file and no built blocks found.")
                return
            mx.logger.info("ℹ️ Blocks not found, will need to build first")
            return

        self.load()

        mode_label = mode.capitalize()
        if force_llm:
            mode_label += "_LLM"
        mx.logger.info("ℹ️ %s mode retrieval will append results to: %s, %s", mode_label, result_jsonl, result_csv)

        # Create trace file for filtering analysis
        trace_file = result_jsonl.replace(".jsonl", "_trace.jsonl")

        header_written = os.path.exists(result_csv)
        os.makedirs(os.path.dirname(result_csv), exist_ok=True)

        csv_file = open(result_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if not header_written:
            writer.writerow([
                "User_ID",
                "QA_ID",
                "Question",
                "Category",
                "Ranking_ContentEventTopicKW",
                "Targets",
                "Mode",
            ])

        # Load QA data from qa file
        # Extract user_id from run_id (format: prefix-userid or just userid)
        user_id_part = mx.Config.RUN_ID.split('-')[-1] if '-' in mx.Config.RUN_ID else mx.Config.RUN_ID
        qa_file = f"data/qa_{user_id_part}.json"
        if not os.path.exists(qa_file):
            mx.logger.error("❌ QA file not found: %s", qa_file)
            return

        with open(qa_file, "r", encoding="utf-8") as f:
            raw_list = json.load(f)[: mx.Config.LIMIT_CONVERSATIONS]

        for data in raw_list:
            user_id = data.get("user_id")
            if user_id is None:
                mx.logger.warning("⚠️ Skipping entry without user_id")
                continue

            for qa_idx, qa in enumerate(data.get("qa", [])):
                if qa.get("category") == 5:
                    continue

                # Add qa_idx to qa dict for ground truth lookup
                qa["qa_idx"] = qa_idx

                rankings, sim_map, target_boxes, filtering_trace = self._score_and_rank(
                    user_id, qa, mode=mode, force_llm=force_llm
                )

                writer.writerow([
                    user_id,
                    qa_idx,
                    qa.get("question", ""),
                    qa.get("category", ""),
                    rankings.get("content_event_topic_kw", []),
                    target_boxes,
                    mode_label,
                ])

                res_entry = {
                    "user_id": user_id,
                    "qa_idx": qa_idx,
                    "question": qa.get("question", ""),
                    "category": qa.get("category", ""),
                    "rankings": rankings,
                    "target_boxes": target_boxes,
                    "mode": mode_label,
                }
                mx.TraceLogger.log(result_jsonl, res_entry)

                # Save filtering trace
                mx.TraceLogger.log(trace_file, filtering_trace)

            mx.logger.info("✅ %s mode retrieval done for user %s", mode_label, user_id)

        csv_file.close()
        mx.logger.info("✅ %s mode retrieval results appended to %s", mode_label, result_jsonl)
        mx.logger.info("✅ Filtering traces saved to %s", trace_file)


def main():
    """Main entry point for running ablation study."""
    import argparse
    mx = _mx()

    parser = argparse.ArgumentParser(description="Run retrieval with different filtering modes")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "parse_only", "time_only", "time_event", "event_only", "full", "all"],
        default="full",
        help="Filtering mode to run (or 'all' to run all modes)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID for output directory"
    )
    parser.add_argument(
        "--raw-data-file",
        type=str,
        default=None,
        help="Raw data file path (JSON with QA pairs)"
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Force LLM parsing (disable fast path) for better accuracy"
    )

    args = parser.parse_args()

    # Apply configuration
    mx.Config.apply_run_id(args.run_id)

    # Set raw data file if provided
    if args.raw_data_file:
        mx.Config.RAW_DATA_FILE = args.raw_data_file

    # Setup logging with file output
    log_file = os.path.join(mx.Config.OUTPUT_DIR, "compare_retrieval_methods.log")
    mx.setup_logging(log_file)

    if not (mx.Config.API_KEY or "").strip():
        mx.logger.warning("⚠️  OPENAI_API_KEY missing; LLM calls may fail.")

    # Initialize worker
    worker = mx.LLMWorker()

    mx.logger.info("ℹ️ Using run_id=%s, output_dir=%s", mx.Config.RUN_ID, mx.Config.OUTPUT_DIR)
    mx.logger.info("ℹ️ Raw data file: %s", mx.Config.RAW_DATA_FILE)
    mx.logger.info("ℹ️ Log file: %s", log_file)

    # Initialize retriever
    retriever = EnhancedRetriever(worker)

    # Run modes
    modes_to_run = ["baseline", "time_only", "time_event", "full"] if args.mode == "all" else [args.mode]

    for mode in modes_to_run:
        mx.logger.info("=" * 80)
        mx.logger.info("Running mode: %s", mode)
        mx.logger.info("=" * 80)

        result_jsonl = os.path.join(mx.Config.OUTPUT_DIR, f"retrieval_{mode}.jsonl")
        result_csv = os.path.join(mx.Config.OUTPUT_DIR, f"retrieval_{mode}.csv")
        trace_file = result_jsonl.replace(".jsonl", "_trace.jsonl")

        # Remove existing files to avoid appending
        for file_path in [result_jsonl, result_csv, trace_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

        t_start = time.perf_counter()
        retriever.run(result_jsonl, result_csv, mode=mode, force_llm=args.force_llm)
        elapsed = time.perf_counter() - t_start
        mx.logger.info("⏱️ Mode [%s] retrieval time: %.2f s", mode, elapsed)

    mx.logger.info("=" * 80)
    mx.logger.info("✅ All modes completed!")
    mx.logger.info("=" * 80)


if __name__ == "__main__":
    main()
