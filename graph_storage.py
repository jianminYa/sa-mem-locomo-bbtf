"""Non-graph compatibility shims for LoCoMo B / B+TF reproduction.

The public repository's `build_impl_graph.py` imports graph helpers at module
load time. B and B+TF do not enable graph build/retrieval, but Python still
needs these symbols to import the builder. Replace this file with the author's
real graph implementation before running graph/HTM experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GraphConfig:
    enable_graph: bool = False
    memgraph_url: str = "bolt://localhost:7687"
    memgraph_username: str = ""
    memgraph_password: str = ""
    graph_similarity_threshold: float = 0.8
    graph_similar_limit: int = 20


class MemgraphEventGraph:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Graph support is not bundled in this non-graph B/B+TF package. "
            "Run without --enable-graph/--graph-expand, or replace graph_storage.py "
            "with the author's full implementation."
        )


def build_graph_payload_from_memblock(*, block: Dict[str, Any], event: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "block_id": block.get("block_id"),
        "event_id": event.get("event_id"),
        "description": event.get("description"),
        "time_metadata": event.get("time_metadata", {}),
    }


def extract_and_store_event_entities(*args: Any, **kwargs: Any) -> Dict[str, int]:
    return {"entity_tool_calls": 0, "relation_tool_calls": 0, "entity_count": 0, "relation_count": 0}


def extract_and_store_block_entities(*args: Any, **kwargs: Any) -> Dict[str, int]:
    return {"entity_tool_calls": 0, "relation_tool_calls": 0, "entity_count": 0, "relation_count": 0}
