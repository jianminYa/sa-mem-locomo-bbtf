"""Non-graph compatibility shims for LoCoMo B / B+TF reproduction.

These classes satisfy imports in `build_impl_graph.py` when graph is disabled.
They are intentionally not a graph implementation.
"""

from __future__ import annotations

from typing import Any


class OpenAIToolCaller:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class OllamaToolCaller:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class GraphEntitiesExtractor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
