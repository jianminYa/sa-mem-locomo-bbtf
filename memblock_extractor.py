import argparse
import json
import os
import logging
import csv
import re
import pathlib
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import wait, FIRST_COMPLETED
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Iterable

import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from build_prompts_halumem import PROMPT_MSG_CONTINUATION, PROMPT_DIALOG_EXTRACT,PROMPT_DIALOG_CLASSIFICATION
from generate_prompts import PROMPT_QA_ANSWER as PROMPT_QA_ANSWER_GEN
from temporal_resolution_tool import resolve_temporal_expression, TEMPORAL_RESOLUTION_FUNCTION_SCHEMA
import sys
if __name__ == "__main__":
    sys.modules.setdefault("memblock_extractor", sys.modules[__name__])
def _event_line_to_obj(line: str) -> Dict[str, str]:
    """Convert an event line like "[start, end] desc" into a structured object."""
    s = str(line or "").strip()
    m = re.match(r"^\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]\s*(.*)$", s)
    if not m:
        return {"start_time": "Unknown", "end_time": "Unknown", "description": s}
    start = str(m.group(1) or "").strip() or "Unknown"
    end = str(m.group(2) or "").strip() or start
    desc = str(m.group(3) or "").strip() or "event"
    return {"start_time": start, "end_time": end, "description": desc}


def _event_any_to_line(ev: Any) -> str:
    """Best-effort convert event (dict or str) into a "[start, end] desc" line."""
    if isinstance(ev, dict):
        start = str(ev.get("start_time", "Unknown") or "Unknown")
        end = str(ev.get("end_time", start) or start)
        desc = str(ev.get("description", "") or "").strip()
        if not desc:
            desc = "event"
        return f"[{start}, {end}] {desc}".strip()
    return str(ev or "").strip()


def _events_to_text(events: Any) -> str:
    """Flatten events (dict list or str list) into a short text blob."""
    if not events:
        return ""
    if isinstance(events, list):
        parts: List[str] = []
        for ev in events:
            if isinstance(ev, dict):
                desc = str(ev.get("description", "") or "").strip()
                if desc:
                    parts.append(desc)
            else:
                s = str(ev or "").strip()
                if s:
                    parts.append(s)
        return " | ".join(parts)
    return str(events).strip()


def _get_block_id(obj: Dict[str, Any]) -> int:
    """Get stable block id from either new (block_id) or legacy (box_id) schema."""
    try:
        v = obj.get("block_id")
        if isinstance(v, int):
            return v
        v2 = obj.get("box_id")
        if isinstance(v2, int):
            return v2
    except Exception:
        pass
    return 0


def _get_block_event_start_time(obj: Dict[str, Any]) -> str:
    try:
        v = obj.get("block_event_start_time")
        if v:
            return str(v)
        v2 = obj.get("box_event_start_time")
        if v2:
            return str(v2)
        v3 = obj.get("block_event_end_time") or obj.get("box_event_end_time")
        return str(v3) if v3 else "Unknown"
    except Exception:
        return "Unknown"


def _get_block_event_end_time(obj: Dict[str, Any]) -> str:
    try:
        v = obj.get("block_event_end_time")
        if v:
            return str(v)
        v2 = obj.get("box_event_end_time")
        if v2:
            return str(v2)
        v3 = obj.get("block_event_start_time") or obj.get("box_event_start_time")
        return str(v3) if v3 else "Unknown"
    except Exception:
        return "Unknown"


def _try_load_dotenv(dotenv_path: str = ".env") -> None:
    """Best-effort .env loader (no external dependency).

    Only fills keys that are missing from the current process environment.
    Supports simple KEY=VALUE lines with optional single/double quotes.
    """
    try:
        if not os.path.exists(dotenv_path):
            return
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                if key in os.environ and os.environ[key].strip():
                    continue
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                os.environ[key] = val
    except Exception:
        return


_try_load_dotenv()


# ================= 主题分类（topic_kw_text -> topic_category） =================
DEFAULT_TOPICS = [
    "personal_details",
    "family",
    "professional_details",
    "sports",
    "travel",
    "food",
    "music",
    "health",
    "technology",
    "hobbies",
    "fashion",
    "entertainment",
    "milestones",
    "user_preferences",
    "misc",
]


def to_iso8601_locomo(t: str) -> str | None:
    """Parse LoCoMo-style datetime strings into ISO8601.

    Examples:
    - "1:56 pm on 8 May, 2023" -> "2023-05-08T13:56:00"
    - "8 May, 2023" -> "2023-05-08"
    Returns None when input is not a recognized LoCoMo format.
    """
    s = (t or "").strip()
    if not s or s.lower() == "unknown":
        return None

    s_norm = re.sub(r"\s+", " ", s).strip()
    s_norm = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), s_norm, flags=re.IGNORECASE)

    fmts = (
        "%I:%M %p on %d %B, %Y",  # 1:56 PM on 8 May, 2023
        "%I:%M %p on %d %b, %Y",  # 1:56 PM on 8 May, 2023 (abbr month)
        "%d %B, %Y",              # 8 May, 2023
        "%d %b, %Y",              # 8 May, 2023 (abbr month)
    )

    for fmt in fmts:
        try:
            dt = datetime.strptime(s_norm, fmt)
            has_time = ("%H" in fmt) or ("%I" in fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S") if has_time else dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def to_iso8601_if_possible(t: str) -> str:
    """Convert dataset time like 'Sep 04, 2025, 11:57:31' -> '2025-09-04T11:57:31'.
    Keep ISO-like strings as-is. Return 'Unknown' if empty.
    """
    s = (t or "").strip()
    if not s or s.lower() == "unknown":
        return "Unknown"

    # already ISO-ish (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS...)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s

    # LoCoMo datetime/date format branch
    locomo_iso = to_iso8601_locomo(s)
    if locomo_iso is not None:
        return locomo_iso

    # LongMemEval-style format: "YYYY/MM/DD (Dow) HH:MM" or date-only
    try:
        s2 = re.sub(r"\s*\([^\)]*\)\s*", " ", s).strip()
        s2 = re.sub(r"\s+", " ", s2)
        for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(s2, fmt)
                has_time = "%H" in fmt
                return dt.strftime("%Y-%m-%dT%H:%M:%S") if has_time else dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    except Exception:
        pass

    # Try parse common dataset formats
    for fmt in ("%b %d, %Y, %H:%M:%S", "%b %d, %Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S") if ":%S" in fmt else dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Fallback: try your existing parser (if present)
    try:
        dt, _ = _parse_event_time_range(f"[{s}, {s}] x")
        if dt is not None:
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass

    return s


def infer_granularity(iso_time: str) -> str:
    s = (iso_time or "").strip()
    if not s or s == "Unknown":
        return "unknown"
    return "second" if "T" in s else "day"


def infer_event_type(description: str) -> str:
    """
    Heuristic event type classifier based on linguistic patterns.

    Rules:
    1. OCCURRENCE: One-time actions (happened/completed/changed)
    2. STATE: Ongoing changeable states (be/have + role/preference/habit)
    3. ATTRIBUTE: Stable identity facts (name/MBTI/nationality/birth date)
    4. INTENTION: Future-oriented plans/goals

    Priority: INTENTION > ATTRIBUTE > OCCURRENCE > STATE
    """
    s = (description or "").strip().lower()

    # 1. INTENTION: Future-oriented plans (highest priority)
    intention_markers = (
        "plans to ", "plan to ",
        "will ", "aims to ", "aim to ",
        "wants to ", "want to ",
        "hopes to ", "hope to ",
        "intends to ", "intend to ",
        "going to ", "preparing to ",
    )
    if any(m in s for m in intention_markers):
        return "INTENTION"

    # 2. ATTRIBUTE: Stable identity facts (check before OCCURRENCE)
    # These are timeless structural information
    attribute_markers = (
        "name is ", "name: ",
        "mbti is ", "mbti: ",
        "nationality is ", "nationality: ",
        "gender is ", "gender: ",
        "birth date is ", "birthday is ",
        "blood type is ", "blood type: ",
        "ethnicity is ", "ethnicity: ",
    )
    if any(m in s for m in attribute_markers):
        return "ATTRIBUTE"

    # 3. OCCURRENCE: One-time events (state changes, completed actions)
    # Priority: Any state change is OCCURRENCE
    occurrence_markers = (
        "was born ", "got married", "became ",
        "started ", "stopped ", "finished ",
        "moved to ", "graduated ", "died ",
        "divorced ", "quit ", "joined ",
        "bought ", "sold ", "visited ",
        "met ", "left ", "arrived ",
        "got ", "received ", "won ",
        "lost ", "broke up", "retired ",
    )
    if any(m in s for m in occurrence_markers):
        return "OCCURRENCE"

    # 4. STATE: Ongoing changeable states (lowest priority)
    # Uses be/have to describe roles, preferences, habits
    # Note: Age is always STATE (e.g., "is 30 years old")
    state_markers = (
        "is married", "is single", "is divorced",
        "works as ", "works at ",
        "lives in ", "lives at ",
        "likes ", "dislikes ", "prefers ",
        "doesn't like ", "does not like ",
        "has a ", "has been ",
        "is a ", "is an ",
        " years old", " year old",  # Age indicators
    )
    if any(m in s for m in state_markers):
        return "STATE"

    # Default: If uncertain, default to OCCURRENCE
    # (Rule 9: When hesitating between OCCURRENCE and ATTRIBUTE, prioritize OCCURRENCE)
    return "OCCURRENCE"

def _normalize_topic_label(label: str) -> str:
    s = (label or "").strip().lower()
    s = re.sub(r"[^a-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _extract_topic_label(text: str) -> str:
    t = _normalize_topic_label(text)
    if t in DEFAULT_TOPICS:
        return t

    lowered = (text or "").lower()
    for cand in DEFAULT_TOPICS:
        if cand in lowered:
            return cand

    return "misc"


def _heuristic_topic_label(topic_kw_text: str) -> str:
    # (keep your existing heuristic implementation unchanged)
    s = (topic_kw_text or "").lower()
    # ...existing code...
    return "misc"


_TOPIC_CACHE_LOCK = threading.Lock()
_TOPIC_CACHE_PATH: str | None = None
_TOPIC_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_topic_cache_if_needed(path: str) -> None:
    global _TOPIC_CACHE_PATH, _TOPIC_CACHE
    with _TOPIC_CACHE_LOCK:
        if _TOPIC_CACHE_PATH == path and _TOPIC_CACHE:
            return
        _TOPIC_CACHE_PATH = path
        _TOPIC_CACHE = {}
        try:
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    # Backward compatible: allow old cache values as string label
                    for k, v in obj.items():
                        key = str(k).strip()
                        if not key:
                            continue
                        if isinstance(v, str):
                            _TOPIC_CACHE[key] = {"topic": _extract_topic_label(v), "confidence": None}
                        elif isinstance(v, dict):
                            lab = _extract_topic_label(str(v.get("topic", "") or "misc"))
                            conf = v.get("confidence", None)
                            try:
                                conf = float(conf) if conf is not None else None
                            except Exception:
                                conf = None
                            _TOPIC_CACHE[key] = {"topic": lab, "confidence": conf}
        except Exception:
            _TOPIC_CACHE = {}


def _save_topic_cache_atomic(path: str, cache: Dict[str, Dict[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ================= 1. 全局配置 =================
class Config:
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

    # Prefer environment variables; can also be overridden by CLI flags.
    API_KEY = os.environ.get("OPENAI_API_KEY", "sk-xx")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "")

    RAW_DATA_FILE = "/data/locomo/data/locomo10.json"
    OUTPUT_BASE_DIR = "out"
    RUN_ID: str | None = "locomo"

    # 路径会在 apply_run_id 时按 run_id 重写
    OUTPUT_DIR = OUTPUT_BASE_DIR
    # Use reindexed file with conversation-level user_ids for LoCoMo
    FINAL_CONTENT_FILE = os.path.join(OUTPUT_DIR, "final_boxes_content.jsonl")
    VECTOR_DIR = os.path.join(OUTPUT_DIR, "vector_store")
    SIMPLE_RETRIEVAL_JSONL = os.path.join(OUTPUT_DIR, "simple_retrieval.jsonl")
    SIMPLE_RETRIEVAL_CSV = os.path.join(OUTPUT_DIR, "simple_retrieval.csv")
    GENERATION_RESULT_FILE = os.path.join(OUTPUT_DIR, "generation_results.jsonl")
    GENERATION_REPORT_CSV = os.path.join(OUTPUT_DIR, "report_generation_qa.csv")
    TOKEN_LOG_FILE = os.path.join(OUTPUT_DIR, "token_stream.jsonl")
    BUILD_CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "build_checkpoint.json")
    BUILD_STATS_FILE = os.path.join(OUTPUT_DIR, "build_stats.jsonl")
    GEN_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "generation_metrics_summary.jsonl")
    TOPIC_CACHE_FILE = os.path.join(OUTPUT_DIR, "topic_cache.json")
    TOPIC_CLASSIFY_TIMEOUT = 60.0
    TOPIC_CLASSIFY_MAX_RETRIES = 4
    EVENT_LABEL_TOP_K = 3
    LOCOMO_SESSION_PREFIX = False
    GRAPH_EXTRACT_SOURCE = "event"

    BUILD_TRACE_FILE = os.path.join(OUTPUT_DIR, "trace_build_process.jsonl")
    TIME_TRACE_FILE = os.path.join(OUTPUT_DIR, "time_traces.jsonl")
    TRACE_PROMPT_LOG_FILE = os.path.join(OUTPUT_DIR, "trace_prompts.jsonl")
    TRACE_METRICS = ["temporal"]

    OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "0") or 0)
    OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "0") or 0)

    USE_GRAPH_CONTEXT = False
    GRAPH_CONTEXT_EVENTS_ONLY = False
    GRAPH_CONTEXT_RELATIONS_ONLY = False
    GRAPH_CONTEXT_EXPANDED_TOPK = None
    GRAPH_CONTEXT_EXPANDED_MIN_SCORE = None
    GRAPH_CONTEXT_PERSON_PROFILE = False
    GRAPH_CONTEXT_STYLE = "default"
    GRAPH_CONTEXT_CATEGORIES = None

    # Event classification control
    # Set to False to skip Pass 2 LLM call (PROMPT_DIALOG_CLASSIFICATION) and use heuristic fallback
    # This saves 1 LLM call per block and reduces build time/cost
    ENABLE_EVENT_CLASSIFICATION = True  # Disable by default for cost savings

    LIMIT_CONVERSATIONS = 2
    LIMIT_SESSIONS = None  # None 表示不限制
    TOP_K_RETRIEVE = 20
    TOP_K_GENERATE = 5
    BUILD_PREV_MSGS = 2
    CHECKPOINT_EVERY_SAMPLE = True

    # Confidence thresholding for no-memory gate
    # If max similarity score < threshold, return empty results (refuse to answer)
    # Set to None to disable confidence thresholding
    CONFIDENCE_THRESHOLD = None  # Default: disabled (always return results)
    # Recommended starting value: 0.3-0.5 (tune based on TNR/Recall tradeoff)

    # 生成阶段使用的 box 数量（答案上下文 TopN）
    # 设置为20以与HaluMem基准保持一致（使用全部检索到的记忆块）
    ANSWER_TOP_N = 20
    # 默认仅运行 content 文本模式
    GEN_TEXT_MODES = ["content"]

    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "gpt-oss:120b")
    OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")

    @classmethod
    def effective_llm_model(cls) -> str:
        return cls.OLLAMA_LLM_MODEL if cls.LLM_PROVIDER == "ollama" else cls.LLM_MODEL

    @classmethod
    def effective_embedding_model(cls) -> str:
        return cls.OLLAMA_EMBEDDING_MODEL if cls.LLM_PROVIDER == "ollama" else cls.EMBEDDING_MODEL

    @classmethod
    def effective_base_url(cls) -> str:
        return cls.OLLAMA_BASE_URL if cls.LLM_PROVIDER == "ollama" else cls.BASE_URL

    @classmethod
    def effective_api_key(cls) -> str:
        return cls.OLLAMA_API_KEY if cls.LLM_PROVIDER == "ollama" else cls.API_KEY

    PROMPT_MSG_CONTINUATION = PROMPT_MSG_CONTINUATION
    PROMPT_DIALOG_EXTRACT = PROMPT_DIALOG_EXTRACT
    PROMPT_DIALOG_CLASSIFICATION = PROMPT_DIALOG_CLASSIFICATION


    PROMPT_QA_ANSWER = PROMPT_QA_ANSWER_GEN

    @classmethod
    def sanitize_run_id(cls, run_id: str | None) -> str:
        rid = (run_id or cls.effective_llm_model() or "default").strip()
        rid = re.sub(r"[^A-Za-z0-9_.-]+", "_", rid)
        return rid or "default"

    @classmethod
    def apply_run_id(cls, run_id: str | None):
        rid = cls.sanitize_run_id(run_id)
        cls.RUN_ID = rid
        cls.OUTPUT_DIR = os.path.join(cls.OUTPUT_BASE_DIR, rid)
        # Use reindexed file with conversation-level user_ids for LoCoMo
        cls.FINAL_CONTENT_FILE = os.path.join(cls.OUTPUT_DIR, "final_boxes_content.jsonl")
        cls.VECTOR_DIR = os.path.join(cls.OUTPUT_DIR, "vector_store")
        cls.SIMPLE_RETRIEVAL_JSONL = os.path.join(cls.OUTPUT_DIR, "simple_retrieval.jsonl")
        cls.SIMPLE_RETRIEVAL_CSV = os.path.join(cls.OUTPUT_DIR, "simple_retrieval.csv")
        cls.GENERATION_RESULT_FILE = os.path.join(cls.OUTPUT_DIR, "generation_results.jsonl")
        cls.GENERATION_REPORT_CSV = os.path.join(cls.OUTPUT_DIR, "report_generation_qa.csv")
        cls.TOKEN_LOG_FILE = os.path.join(cls.OUTPUT_DIR, "token_stream.jsonl")
        cls.BUILD_TRACE_FILE = os.path.join(cls.OUTPUT_DIR, "trace_build_process.jsonl")
        cls.TIME_TRACE_FILE = os.path.join(cls.OUTPUT_DIR, "time_traces.jsonl")
        cls.TRACE_PROMPT_LOG_FILE = os.path.join(cls.OUTPUT_DIR, "trace_prompts.jsonl")
        cls.BUILD_STATS_FILE = os.path.join(cls.OUTPUT_DIR, "build_stats.jsonl")
        cls.GEN_SUMMARY_FILE = os.path.join(cls.OUTPUT_DIR, "generation_metrics_summary.jsonl")
        cls.BUILD_CHECKPOINT_FILE = os.path.join(cls.OUTPUT_DIR, "build_checkpoint.json")
        cls.TOPIC_CACHE_FILE = os.path.join(cls.OUTPUT_DIR, "topic_cache.json")
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_DIR, exist_ok=True)

    @classmethod
    def vector_file(cls, user_id: Any) -> str:
        return os.path.join(cls.VECTOR_DIR, f"user_{user_id}.json")


def _tqdm(iterable: Iterable, total: int | None = None, desc: str | None = None, **kwargs):
    """Optional progress bar (tqdm if installed; otherwise passthrough)."""
    try:
        # Avoid progress bar conflicts in parallel workers.
        if threading.current_thread() is not threading.main_thread():
            return iterable
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, **kwargs)
    except Exception:
        return iterable


def _apply_uuid_user_ids(
    boxes: List[Dict[str, Any]],
    uuid8: str,
    user_id_start: int,
    user_count: int,
) -> None:
    """Rewrite numeric user_id(s) produced by build_all to uuid-based ids.

    - If user_count == 1: user_id = uuid8
    - Else: user_id = f"{uuid8}_{i}" for each sample index i
    """

    if user_count <= 0:
        return
    mapping: Dict[int, str] = {}
    for i in range(user_count):
        mapping[user_id_start + i] = uuid8 if user_count == 1 else f"{uuid8}_{i}"

    for b in boxes:
        try:
            old = b.get("user_id")
            if isinstance(old, int) and old in mapping:
                b["user_id"] = mapping[old]
        except Exception:
            continue


def _list_input_files(raw_data_dir: str, pattern: str = "*.json") -> List[str]:
    p = pathlib.Path(raw_data_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"raw-data-dir not found or not a directory: {raw_data_dir}")
    files = sorted([str(x) for x in p.glob(pattern) if x.is_file()])
    return files


def _load_checkpoint(path: str) -> Dict[str, Any] | None:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _save_checkpoint(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def setup_logging(log_file: str = None):
    """
    Setup logging with optional file output.

    Args:
        log_file: Path to log file. If None, only console logging is used.
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Initialize logging (console only by default)
setup_logging()
logger = logging.getLogger(__name__)

_APPEND_LOCK = threading.Lock()

# Thread-local context for per-worker build metadata.
# Used to map numeric user_id (0,1,2,...) to stable uuid-based ids (e.g., ab12cd34, ab12cd34_1)
# at log time (TraceLogger), so trace files don't leak numeric ids in parallel mode.
_THREAD_CTX = threading.local()


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _load_raw_conversations(path: str) -> List[Dict[str, Any]]:
    """Load raw conversations as a list.

    Supports either:
    - JSON list file:  [ {"conversation": {...}}, ... ]
    - JSONL file:      one JSON object per line
    """
    if not path:
        raise ValueError("RAW data file path is empty")

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    # Peek first non-empty bytes to decide JSON vs JSONL.
    with p.open("rb") as fb:
        head = fb.read(4096)
    head_stripped = head.lstrip()
    is_json_list = head_stripped.startswith(b"[")

    if is_json_list:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON list at top-level")
        return data

    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


# ================= 2. 基础服务类 =================
class TraceLogger:
    @staticmethod
    def log(file_path, data):
        # In parallel build, we intentionally assign numeric user_id (0,1,2,...) inside each worker.
        # Map those to stable uuid-based ids at write time so log files stay consistent.
        data_to_write = data
        try:
            user_id_map = getattr(_THREAD_CTX, "user_id_map", None)
            if user_id_map and isinstance(data, dict) and "user_id" in data:
                sid = data.get("user_id")
                sid_int = None
                if isinstance(sid, int):
                    sid_int = sid
                elif isinstance(sid, str) and sid.isdigit():
                    sid_int = int(sid)
                if sid_int is not None and sid_int in user_id_map:
                    data_to_write = dict(data)
                    data_to_write["user_id"] = user_id_map[sid_int]
        except Exception:
            data_to_write = data
        with _APPEND_LOCK:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_to_write, ensure_ascii=False) + "\n")


class TokenAnalyzer:
    stage_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})

    @staticmethod
    def log_usage(usage, note, extra=None):
        if not usage:
            return
        stage = None
        if extra and "stage" in extra:
            stage = extra["stage"]
        entry = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "note": note,
            "in": usage.prompt_tokens,
            "out": getattr(usage, "completion_tokens", 0),
        }
        if extra:
            entry.update(extra)
        with _APPEND_LOCK:
            with open(Config.TOKEN_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            if stage:
                stats = TokenAnalyzer.stage_stats[stage]
                stats["calls"] += 1
                stats["prompt"] += usage.prompt_tokens
                stats["completion"] += getattr(usage, "completion_tokens", 0)
                stats["total"] += getattr(usage, "total_tokens", usage.prompt_tokens + getattr(usage, "completion_tokens", 0))

    @staticmethod
    def get_stage_stats(stage: str) -> Dict[str, float]:
        return TokenAnalyzer.stage_stats.get(stage, {"calls": 0, "prompt": 0, "completion": 0, "total": 0})


def evidence_to_targets(evidence_list, boxes):
    """Map evidence like 'D1:3' to box_ids covering that session/message."""
    targets = set()
    if not evidence_list:
        return []

    session_map = {}
    for b in boxes:
        cov = b.get("coverage", {})
        raw_sid = cov.get("session_id")
        sid_norm = None
        try:
            if isinstance(raw_sid, str) and raw_sid.startswith("session_"):
                sid_norm = int(raw_sid.split("_")[1])
            else:
                sid_norm = int(raw_sid)
        except Exception:
            sid_norm = raw_sid
        session_map.setdefault(sid_norm, []).append(b)

    for ev in evidence_list:
        try:
            part = ev.split(":")
            sid = int(part[0][1:])
            msg_idx = int(part[1])
            if sid not in session_map:
                continue
            for b in session_map[sid]:
                cov = b.get("coverage", {})
                if cov.get("start_idx", 0) <= msg_idx <= cov.get("end_idx", 0):
                    targets.add(_get_block_id(b))
        except Exception:
            continue

    return sorted(list(targets))


class EmbeddingStore:
    """Lazy embedding cache: per-sample load, fetch-or-compute, flush on demand."""

    def __init__(self, worker: "LLMWorker", user_id: Any):
        self.worker = worker
        self.user_id = user_id
        self.path = Config.vector_file(user_id)
        self.data: Dict[str, Dict[str, Any]] = {}
        self.dirty = False
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
                logging.getLogger(__name__).info(
                    "🔍 VectorStore file loaded path=%s keys=%d",
                    self.path,
                    len(self.data),
                )
            except Exception as e:
                self.data = {}
                logging.getLogger(__name__).warning(
                    "🔍 VectorStore file parse failed path=%s err=%s",
                    self.path,
                    str(e),
                )
        else:
            logging.getLogger(__name__).warning(
                "🔍 VectorStore file missing path=%s",
                self.path,
            )

    def get_vector(self, key: str, field: str, text: str, note: str, stage: str | None = None) -> List[float]:
        if not text:
            return []
        if key in self.data and field in self.data[key]:
            return self.data[key][field]
        vec = self.worker.get_embedding(text, note=note)
        self.data.setdefault(key, {})[field] = vec
        self.dirty = True
        return vec

    def ensure_key(self, key: str):
        if key not in self.data:
            self.data[key] = {}
            self.dirty = True

    def flush(self):
        if not self.dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)
        self.dirty = False


class LLMWorker:
    def __init__(self):
        self.provider = (Config.LLM_PROVIDER or "openai").strip().lower()
        self.base_url = Config.effective_base_url()
        self.api_key = Config.effective_api_key()
        self.chat_model = Config.effective_llm_model()
        self.embedding_model = Config.effective_embedding_model()
        self.embedding_fallback_dims = 1536

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        try:
            self.encoding = tiktoken.encoding_for_model(self.chat_model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    def _create_chat_completion(self, *, messages: List[Dict[str, str]], json_mode: bool = False, timeout: float | None = None):
        kwargs: Dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.0,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception:
            if json_mode:
                kwargs.pop("response_format", None)
                return self.client.chat.completions.create(**kwargs)
            raise

    def get_embedding(self, text, note="Emb"):
        try:
            if not text:
                return [0.0] * self.embedding_fallback_dims
            resp = self.client.embeddings.create(
                input=text.replace("\n", " "), model=self.embedding_model
            )
            emb = None
            try:
                emb = resp.data[0].embedding
            except Exception:
                emb = None
            return emb if emb is not None else [0.0] * self.embedding_fallback_dims
        except Exception:
            return [0.0] * self.embedding_fallback_dims

    def chat_completion(self, prompt, note="Completion", json_mode=False, extra=None, enable_functions=False):
        """
        Call LLM with optional function calling support.

        Args:
            prompt: The prompt text
            note: Note for token logging
            json_mode: Whether to use JSON response format
            extra: Extra metadata for logging
            enable_functions: Whether to enable temporal resolution function calling
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            # Add function calling support for temporal resolution
            kwargs = {
                "model": self.chat_model,
                "messages": messages,
                "temperature": 0.0,
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            if enable_functions:
                kwargs["tools"] = [{
                    "type": "function",
                    "function": TEMPORAL_RESOLUTION_FUNCTION_SCHEMA
                }]
                kwargs["tool_choice"] = "auto"

            resp = self.client.chat.completions.create(**kwargs)

            # Handle function calling
            if enable_functions and resp.choices[0].message.tool_calls:
                # Execute function calls
                tool_calls = resp.choices[0].message.tool_calls
                messages.append(resp.choices[0].message)

                for tool_call in tool_calls:
                    if tool_call.function.name == "resolve_temporal_expression":
                        args = json.loads(tool_call.function.arguments)
                        result = resolve_temporal_expression(
                            args["observation_time"],
                            args["expression"]
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })

                # Get final response after function execution
                kwargs["messages"] = messages
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                resp = self.client.chat.completions.create(**kwargs)

            extra_payload = {"prompt_tokens_est": self.count_tokens(prompt)}
            if extra:
                extra_payload.update(extra)
            TokenAnalyzer.log_usage(resp.usage, note, extra_payload)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return "{}" if json_mode else ""

    def check_relation(self, text_prev_list, text_curr, note="Relation"):
        ref_text = "\n".join(text_prev_list)
        prompt = Config.PROMPT_MSG_CONTINUATION.format(ref=ref_text, curr=text_curr)
        res = self.chat_completion(prompt, note=note, extra={"prompt_tokens_est": self.count_tokens(prompt), "stage": "build"})
        return "Yes" if "yes" in res.lower() else "No"

    def classify_event_labels(
        self,
        event_text: str,
        top_k: int | None = None,
        with_meta: bool = True,
    ) -> List[Dict[str, Any]] | List[str]:
        """
        Classify an event into top_k labels from DEFAULT_TOPICS.

        - If with_meta=True:
          return [{"label": "<one of DEFAULT_TOPICS>", "confidence": 0.0~1.0, ...]
        - If with_meta=False:
          return ["label1", "label2", ...]
        """
        text = (event_text or "").strip()
        k = int(top_k or getattr(Config, "EVENT_LABEL_TOP_K", 3) or 3)
        k = max(1, min(k, len(DEFAULT_TOPICS)))

        if not text:
            out0 = [{"label": "misc", "confidence": 0.0}]
            return out0 if with_meta else ["misc"]

        def _clamp01(x: Any, default: float = 0.5) -> float:
            try:
                v = float(x)
            except Exception:
                v = default
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            return v

        # No API key -> heuristic fallback (single label)
        if not (Config.API_KEY or "").strip():
            try:
                lab = _heuristic_topic_label(text)
            except Exception:
                lab = "misc"
            lab = _extract_topic_label(str(lab or "misc"))
            out1 = [{"label": lab, "confidence": 0.5}]
            return out1 if with_meta else [lab]

        system = (
            "You are a strict multi-label classifier for memory events.\n"
            f"Choose up to {k} labels from the allowed set ONLY.\n"
            f"Allowed labels: {', '.join(DEFAULT_TOPICS)}.\n"
            'Return ONLY JSON in this exact schema:\n'
            '{"labels":[{"label":"<allowed_label>","confidence":0.0}]}\n'
            "Rules:\n"
            "- confidence is a float in [0,1]\n"
            "- sort by descending confidence\n"
            "- no extra keys\n"
        )
        user = f"Event description:\n{text}\n"

        timeout = float(getattr(Config, "TOPIC_CLASSIFY_TIMEOUT", 60.0) or 60.0)
        max_retries = int(getattr(Config, "TOPIC_CLASSIFY_MAX_RETRIES", 4) or 4)

        for attempt in range(max_retries + 1):
            try:
                resp = self._create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    json_mode=True,
                    timeout=timeout,
                )
                content = (resp.choices[0].message.content or "").strip()
                obj = json.loads(content) if content else {}
                raw = obj.get("labels", []) if isinstance(obj, dict) else []

                out: List[Dict[str, Any]] = []
                seen = set()
                if isinstance(raw, list):
                    for it in raw:
                        if not isinstance(it, dict):
                            continue
                        lab = _extract_topic_label(str(it.get("label") or "misc"))
                        if lab in seen:
                            continue
                        seen.add(lab)
                        conf = _clamp01(it.get("confidence", 0.5), default=0.5)
                        out.append({"label": lab, "confidence": conf})
                        if len(out) >= k:
                            break

                if out:
                    return out if with_meta else [x["label"] for x in out]

            except Exception:
                time.sleep(min(0.5 * (2**attempt), 8.0))
                continue

        # degrade
        try:
            lab = _heuristic_topic_label(text)
        except Exception:
            lab = "misc"
        lab = _extract_topic_label(str(lab or "misc"))
        out2 = [{"label": lab, "confidence": 0.5}]
        return out2 if with_meta else [lab]


# ================= 3. 构建 (Build) =================
from build_impl_graph import TopicClusterManager, MemoryBuilder

def _parse_event_time_range(event_line: str) -> Tuple[datetime | None, datetime | None]:
    """Parse event line format: "[start, end] event" into datetimes.

    Accepts ISO8601 (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) and
    the dataset style (e.g., "Sep 04, 2025, 09:27:31").
    Returns (None, None) if not parseable.
    """

    def _extract_dt_candidates(text: str) -> List[str]:
        t = str(text or "")
        # Prefer ISO; else dataset format with commas.
        iso_dt = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?", t)
        if iso_dt:
            return iso_dt
        iso_date = re.findall(r"\d{4}-\d{2}-\d{2}", t)
        if iso_date:
            return iso_date
        ds_dt = re.findall(r"[A-Za-z]{3} \d{2}, \d{4}, \d{2}:\d{2}:\d{2}", t)
        if ds_dt:
            return ds_dt
        ds_date = re.findall(r"[A-Za-z]{3} \d{2}, \d{4}", t)
        return ds_date

    def _parse_datetime_fuzzy(s: str) -> datetime | None:
        s = (s or "").strip()
        s = s.strip('"').strip("'").strip("`")
        if not s or s.lower() == "unknown":
            return None

        # If the string contains an ISO/dataset substring, extract it.
        cands = _extract_dt_candidates(s)
        if cands:
            s = cands[0]

        # ISO datetime with optional timezone (normalize to naive UTC)
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?", s):
                s2 = s
                if s2.endswith("Z"):
                    s2 = s2[:-1] + "+00:00"
                if re.fullmatch(r".*[+-]\d{4}$", s2):
                    s2 = s2[:-5] + s2[-5:-2] + ":" + s2[-2:]
                dt = datetime.fromisoformat(s2)
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt
        except Exception:
            pass

        # ISO date
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            pass

        # Dataset timestamp
        for fmt in ("%b %d, %Y, %H:%M:%S", "%b %d, %Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    s = str(event_line or "").strip()
    if not s:
        return None, None
    s = re.sub(r"^\s*[-*]\s+", "", s)

    # Extract up to 2 datetime/date candidates from inside brackets.
    m1 = re.match(r"^\s*\[\s*([^\]]+)\s*\]\s*(.*)$", s)
    if not m1:
        return None, None
    inner = m1.group(1)
    cands = _extract_dt_candidates(inner)
    if not cands:
        return None, None
    if len(cands) == 1:
        dt = _parse_datetime_fuzzy(cands[0])
        return dt, dt

    start_dt = _parse_datetime_fuzzy(cands[0])
    end_dt = _parse_datetime_fuzzy(cands[1])
    if start_dt is not None and end_dt is None:
        end_dt = start_dt
    if end_dt is not None and start_dt is None:
        start_dt = end_dt
    return start_dt, end_dt

def _coerce_temporal_type(t: Any) -> str:
    s = str(t or "").strip().upper()
    # tolerate legacy/alt names if any
    mapping = {
        "EVENT": "OCCURRENCE",
        "WINDOW": "OCCURRENCE",
        "AS_OF": "STATE",
        "PLAN": "INTENTION",
        "PLANNING": "INTENTION",
    }
    s = mapping.get(s, s)
    return s if s in {"OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"} else ""


def _normalize_event_objs(
    events: List[Any],  # accepts dict or str
    session_start_time: str | None,
    session_end_time: str | None,
    fallback_text: str | None = None,
) -> List[Dict[str, Any]]:
    """Normalize events into dict objects while preserving LLM-provided 'type'."""

    def _to_iso(raw: str | None, is_end: bool = False, prefer_end_of_day: bool = False) -> str | None:
        if not raw:
            return None
        raw2 = str(raw).strip().strip('"').strip("'").strip("`")
        dt, _ = _parse_event_time_range(f"[{raw2}] x")
        if dt is None:
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw2) and (is_end or prefer_end_of_day):
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    default_start = _to_iso(session_start_time, is_end=False) or "Unknown"
    default_end = _to_iso(session_end_time, is_end=True) or default_start

    out: List[Dict[str, Any]] = []
    for ev in events or []:
        if isinstance(ev, dict):
            desc = str(ev.get("description", "") or "").strip() or (fallback_text or "event")
            st_raw = ev.get("start_time", "Unknown")
            en_raw = ev.get("end_time", st_raw)

            st = _to_iso(st_raw, is_end=False) or default_start
            en = _to_iso(en_raw, is_end=True, prefer_end_of_day=True) or default_end

            typ = _coerce_temporal_type(ev.get("type"))
            obj: Dict[str, Any] = {"start_time": st, "end_time": en, "description": desc}
            if typ:
                obj["type"] = typ
            out.append(obj)
        else:
            # fallback: best-effort line -> dict (no type)
            line = str(_event_any_to_line(ev) or "").strip()
            if not line:
                continue
            out.append({"start_time": default_start, "end_time": default_end, "description": line})

    # if nothing extracted, create one fallback record
    if not out and fallback_text:
        out.append({"start_time": default_start, "end_time": default_end, "description": str(fallback_text).strip()})

    return out


def _dedupe_and_filter_event_objs(events: List[Dict[str, Any]], drop_texts: List[str] | None = None) -> List[Dict[str, Any]]:
    drop_set = {str(t).strip() for t in (drop_texts or []) if str(t).strip()}

    def _key(e: Dict[str, Any]) -> str:
        desc = str(e.get("description", "") or "").strip()
        st = str(e.get("start_time", "") or "")
        en = str(e.get("end_time", "") or "")
        typ = str(e.get("type", "") or "")
        return f"{typ}|||{st}|||{en}|||{desc}"

    uniq: List[Dict[str, Any]] = []
    seen = set()
    for e in events or []:
        desc = str(e.get("description", "") or "").strip()
        if not desc or desc in drop_set:
            continue
        k = _key(e)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)

    if len(uniq) <= 1:
        return uniq

    # drop huge keyword-dump summaries if more specific items exist
    filtered: List[Dict[str, Any]] = []
    for e in uniq:
        desc = str(e.get("description", "") or "").strip()
        if len(desc) > 120 and "," in desc and any(len(str(x.get("description", "") or "")) < 120 for x in uniq):
            continue
        filtered.append(e)
    return filtered or uniq
def _normalize_event_lines(
    events: List[Any],  # ✅ accept dict or str
    session_start_time: str | None,
    session_end_time: str | None,
    fallback_text: str | None = None,
) -> List[str]:
    """Ensure event lines conform to "[start, end] event" and fill missing/Unknown time with session window."""

    def _to_iso(raw: str | None, is_end: bool = False, prefer_end_of_day: bool = False) -> str | None:
        if not raw:
            return None
        raw2 = str(raw).strip().strip('"').strip("'").strip("`")
        # Parse raw datetime directly (session window may contain commas).
        dt, _ = _parse_event_time_range(f"[{raw2}] x")
        if dt is None:
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw2) and (is_end or prefer_end_of_day):
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    default_start = _to_iso(session_start_time, is_end=False) or "Unknown"
    default_end = _to_iso(session_end_time, is_end=True) or default_start

    normalized: List[str] = []
    for ev in events or []:
        # ✅ key fix: convert dict/str -> "[start, end] desc" first
        line = _event_any_to_line(ev)
        line = str(line or "").strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*]\s+", "", line)

        # Any bracketed time: extract candidates from inside and normalize.
        m = re.match(r"^\s*\[\s*([^\]]+)\s*\]\s*(.*)$", line)
        if m:
            inner = m.group(1)
            rest = (m.group(2) or "").strip()
            cands = re.findall(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?|\d{4}-\d{2}-\d{2}|[A-Za-z]{3} \d{2}, \d{4}, \d{2}:\d{2}:\d{2}|[A-Za-z]{3} \d{2}, \d{4}",
                inner,
            )
            start_raw = cands[0] if len(cands) >= 1 else "Unknown"
            end_raw = cands[1] if len(cands) >= 2 else start_raw
            start_iso = _to_iso(start_raw, is_end=False) or default_start
            end_iso = _to_iso(end_raw, is_end=True, prefer_end_of_day=True) or default_end
            if not rest:
                rest = fallback_text or "event"
            normalized.append(f"[{start_iso}, {end_iso}] {rest}".strip())
            continue

        normalized.append(f"[{default_start}, {default_end}] {line}".strip())

    if normalized:
        return normalized
    if fallback_text:
        return [f"[{default_start}, {default_end}] {fallback_text}".strip()]
    return []


def _dedupe_and_filter_events(events: List[str], drop_texts: List[str] | None = None) -> List[str]:
    """Remove exact duplicates and drop low-signal 'summary' events when more specific ones exist."""

    def _event_text(line: str) -> str:
        s = str(line or "").strip()
        m = re.match(r"^\s*\[[^\]]*\]\s*(.*)$", s)
        return (m.group(1) if m else s).strip()

    drop_set = {str(t).strip() for t in (drop_texts or []) if str(t).strip()}
    seen = set()
    uniq: List[str] = []
    for ev in events or []:
        e = str(ev or "").strip()
        if not e or e in seen:
            continue
        seen.add(e)
        uniq.append(e)

    if len(uniq) <= 1:
        return uniq

    filtered: List[str] = []
    for ev in uniq:
        txt = _event_text(ev)
        if txt in drop_set:
            continue
        # Heuristic: drop huge keyword-dump summary lines if other events exist.
        if len(txt) > 120 and "," in txt and any(len(_event_text(x)) < 120 for x in uniq):
            continue
        filtered.append(ev)
    return filtered or uniq




def _append_jsonl(dst_path: str, src_path: str) -> int:
    if not os.path.exists(src_path):
        return 0
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    mode = "a" if os.path.exists(dst_path) else "w"
    n = 0
    with open(src_path, "r", encoding="utf-8") as fin:
        with open(dst_path, mode, encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                fout.write(line)
                n += 1
    return n



def _write_boxes_jsonl(path: str, boxes: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")




class TraceLinker:
    """Compatibility placeholder for runs that do not request trace context."""

    def __init__(self, worker: Any, trace_metrics: List[str] | None = None):
        self.worker = worker
        self.trace_metrics = trace_metrics or []

    def run(self):
        os.makedirs(os.path.dirname(Config.TIME_TRACE_FILE), exist_ok=True)
        if not os.path.exists(Config.TIME_TRACE_FILE):
            open(Config.TIME_TRACE_FILE, "w", encoding="utf-8").close()
        logger.info("TraceLinker compatibility stub wrote empty trace file: %s", Config.TIME_TRACE_FILE)


from retrieval.retrieval_impl import SimpleRetriever


from generate_impl import AnswerGenerator

def _announce_outputs(stage: str, paths: List[str]):
    targets = [p for p in paths if p]
    if targets:
        logger.info("ℹ️ Stage %s will modify/append: %s", stage, ", ".join(targets))

def main():
    parser = argparse.ArgumentParser(description="Memory build/retrieve/generate pipeline")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=Config.LLM_PROVIDER,
        help="LLM provider selector: openai (default) or ollama (local).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model name (applies to selected provider).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model name (applies to selected provider).",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="Override Ollama OpenAI-compatible base URL (default: http://localhost:11434/v1).",
    )
    parser.add_argument(
        "--ollama-api-key",
        type=str,
        default=None,
        help="Override Ollama API key (default: ollama).",
    )
    parser.add_argument(
        "--stage",
        choices=["build", "retrieve", "generate", "all"],
        default="all",
        help="Which stage to run; 'all' runs the full pipeline sequentially",
    )
    parser.add_argument("--build-prev-msgs", type=int, default=Config.BUILD_PREV_MSGS, help="How many previous messages to use when deciding splits")
    parser.add_argument("--answer-topn", type=str, default=str(Config.ANSWER_TOP_N), help="Number of boxes to use when answering (comma-separated)")
    parser.add_argument(
        "--text-modes",
        nargs="+",
        choices=["content"],
        help="Text modes for generation; default content only",
    )
    parser.add_argument("--run-id", type=str, help="Run identifier (defaults to model name)")
    parser.add_argument(
        "--raw-data-file",
        type=str,
        default=Config.RAW_DATA_FILE,
        help="Raw data path (JSON list or JSONL).",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help="Directory containing per-uuid JSON files (each a JSON list of samples).",
    )
    parser.add_argument(
        "--raw-data-glob",
        type=str,
        default="*.json",
        help="Glob pattern inside raw-data-dir (default: *.json).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for raw-data-dir build (default: 1).",
    )
    parser.add_argument(
        "--uuid-start",
        type=int,
        default=None,
        help="Start index in raw-data-dir file list (0-based). If omitted, may resume from checkpoint.",
    )
    parser.add_argument(
        "--uuid-count",
        type=int,
        default=0,
        help="How many uuid files to process from start (0 means all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume build from build checkpoint if present (raw-data-dir mode).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume even if checkpoint exists.",
    )
    parser.add_argument(
        "--show-checkpoint",
        action="store_true",
        help="Print build checkpoint (if exists) and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List raw-data-dir files and planned slice, then exit.",
    )
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=Config.LIMIT_CONVERSATIONS if Config.LIMIT_CONVERSATIONS is not None else -1,
        help="Limit number of conversations to process (-1 means no limit).",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=Config.LIMIT_SESSIONS if Config.LIMIT_SESSIONS is not None else -1,
        help="Limit number of sessions per conversation (-1 means no limit).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Override API key (or set OPENAI_API_KEY).")
    parser.add_argument("--base-url", type=str, default=None, help="Override base URL (or set OPENAI_BASE_URL).")
    args = parser.parse_args()

    Config.LLM_PROVIDER = (args.provider or Config.LLM_PROVIDER or "openai").strip().lower()

    if args.ollama_base_url is not None:
        Config.OLLAMA_BASE_URL = args.ollama_base_url
    if args.ollama_api_key is not None:
        Config.OLLAMA_API_KEY = args.ollama_api_key

    if args.llm_model is not None:
        if Config.LLM_PROVIDER == "ollama":
            Config.OLLAMA_LLM_MODEL = args.llm_model
        else:
            Config.LLM_MODEL = args.llm_model

    if args.embedding_model is not None:
        if Config.LLM_PROVIDER == "ollama":
            Config.OLLAMA_EMBEDDING_MODEL = args.embedding_model
        else:
            Config.EMBEDDING_MODEL = args.embedding_model

    Config.apply_run_id(args.run_id)
    Config.BUILD_PREV_MSGS = max(1, args.build_prev_msgs)
    Config.TOP_K_RETRIEVE = None  # Keep full ranking

    Config.RAW_DATA_FILE = args.raw_data_file
    Config.LIMIT_CONVERSATIONS = None if args.limit_conversations == -1 else max(0, args.limit_conversations)
    Config.LIMIT_SESSIONS = None if args.limit_sessions == -1 else max(0, args.limit_sessions)
    if args.api_key is not None:
        if Config.LLM_PROVIDER == "ollama":
            Config.OLLAMA_API_KEY = args.api_key
        else:
            Config.API_KEY = args.api_key
    if args.base_url is not None:
        if Config.LLM_PROVIDER == "ollama":
            Config.OLLAMA_BASE_URL = args.base_url
        else:
            Config.BASE_URL = args.base_url
    
    try:
        topn_list = [int(x) for x in args.answer_topn.split(",")]
    except ValueError:
        topn_list = [int(args.answer_topn)]
    Config.ANSWER_TOP_N = topn_list[0]

    if Config.LLM_PROVIDER == "openai" and not (Config.API_KEY or "").strip():
        logger.warning("⚠️  OPENAI_API_KEY missing; LLM calls may fail and outputs may be low-quality.")
    if Config.LLM_PROVIDER == "ollama":
        logger.info(
            "ℹ️ Ollama mode enabled: model=%s, embedding=%s, base_url=%s",
            Config.OLLAMA_LLM_MODEL,
            Config.OLLAMA_EMBEDDING_MODEL,
            Config.OLLAMA_BASE_URL,
        )
    worker = LLMWorker()

    text_modes = args.text_modes or Config.GEN_TEXT_MODES


    logger.info("ℹ️ Using run_id=%s, output_dir=%s", Config.RUN_ID, Config.OUTPUT_DIR)

    if args.show_checkpoint:
        ck = _load_checkpoint(Config.BUILD_CHECKPOINT_FILE)
        if ck is None:
            print("No checkpoint found:", Config.BUILD_CHECKPOINT_FILE)
        else:
            print(json.dumps(ck, ensure_ascii=False, indent=2))
        return

    if args.stage in ("build", "all"):
        _announce_outputs("build", [Config.FINAL_CONTENT_FILE, Config.TOKEN_LOG_FILE, Config.BUILD_STATS_FILE, Config.VECTOR_DIR])
        TokenAnalyzer.stage_stats["build"] = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
        builder = MemoryBuilder(worker)
        if args.raw_data_dir:
            files = _list_input_files(args.raw_data_dir, args.raw_data_glob)
            ck = _load_checkpoint(Config.BUILD_CHECKPOINT_FILE) if (args.resume and not args.no_resume) else None
            default_start = int(ck.get("next_file_index", 0)) if ck else 0
            start_idx = args.uuid_start if args.uuid_start is not None else default_start
            start_idx = max(0, min(start_idx, len(files)))
            count = args.uuid_count
            end_idx = len(files) if not count or count <= 0 else min(len(files), start_idx + count)
            planned = files[start_idx:end_idx]

            if args.dry_run:
                print(f"Found {len(files)} file(s) in {args.raw_data_dir} ({args.raw_data_glob})")
                print(f"Planned slice: start={start_idx} count={count or 'ALL'} end={end_idx} (total={len(planned)})")
                for i, fp in enumerate(planned, start=start_idx):
                    print(f"[{i}] {fp}")
                return

            ck_obj: Dict[str, Any] = ck or {
                "run_id": Config.RUN_ID,
                "raw_data_dir": args.raw_data_dir,
                "raw_data_glob": args.raw_data_glob,
                "created_at": datetime.now().isoformat(),
                "next_file_index": start_idx,
                "processed_files": [],
                "user_id_next": 0,
            }

            processed_prev = set(ck_obj.get("processed_files") or []) if isinstance(ck_obj.get("processed_files"), list) else set()
            tmp_dir = os.path.join(Config.OUTPUT_DIR, "tmp_build")
            os.makedirs(tmp_dir, exist_ok=True)

            def _count_sessions_in_raw_list(raw_list: List[Dict[str, Any]]) -> int:
                if Config.LIMIT_CONVERSATIONS is not None:
                    raw_list = raw_list[: Config.LIMIT_CONVERSATIONS]
                total = 0
                for conversation_data in raw_list:
                    conv = (conversation_data or {}).get("conversation", {})
                    if not isinstance(conv, dict):
                        continue
                    keys = sorted(
                        [k for k in conv.keys() if k.startswith("session_") and len(k) < 12],
                        key=lambda x: int(x.split("_")[1]),
                    )
                    session_keys = keys[: Config.LIMIT_SESSIONS] if Config.LIMIT_SESSIONS is not None else keys
                    total += len(session_keys)
                return total

            def _count_sessions_in_file(file_path: str) -> int:
                try:
                    raw_list = _load_raw_conversations(file_path)
                    return _count_sessions_in_raw_list(raw_list)
                except Exception:
                    return 0

            def _job(file_index: int, file_path: str) -> Tuple[int, str, str, int, Dict[str, Any]]:
                raw_list = _load_raw_conversations(file_path)
                source_id = pathlib.Path(file_path).stem
                uuid8 = (source_id or "")[:8]
                for item in raw_list:
                    if isinstance(item, dict) and "_source_id" not in item:
                        item["_source_id"] = source_id

                local_worker = LLMWorker()
                local_builder = MemoryBuilder(local_worker)

                # Ensure logs in this worker use uuid-based sample ids.
                # (Build itself uses numeric 0..N-1 sample ids within a file.)
                if uuid8:
                    _THREAD_CTX.user_id_map = {
                        i: (uuid8 if i == 0 else f"{uuid8}_{i}") for i in range(len(raw_list))
                    }
                else:
                    _THREAD_CTX.user_id_map = None

                def _on_session_done():
                    # Main thread aggregates session-level progress; workers must not emit tqdm.
                    session_q.put(1)

                try:
                    # Build with numeric user_id_start=0, then rewrite to uuid-based ids.
                    user_id_start = 0
                    boxes = local_builder.build_all(
                        raw_list_override=raw_list,
                        user_id_start=user_id_start,
                        write_incremental=False,
                        on_session_done=_on_session_done,
                    )
                    if uuid8:
                        _apply_uuid_user_ids(boxes, uuid8=uuid8, user_id_start=user_id_start, user_count=len(raw_list))
                    out_tmp = os.path.join(tmp_dir, f"{source_id}.jsonl")
                    _write_boxes_jsonl(out_tmp, boxes)
                    local_stats = {
                        "boxes": local_builder.total_boxes,
                        "messages": sum(local_builder.msg_counts) if local_builder.msg_counts else 0,
                    }
                    return file_index, file_path, out_tmp, len(raw_list), local_stats
                finally:
                    # Avoid leaking mappings between tasks on pooled threads.
                    try:
                        _THREAD_CTX.user_id_map = None
                    except Exception:
                        pass

            # Decide which files to process
            to_process: List[Tuple[int, str]] = []
            to_skip: List[Tuple[int, str]] = []
            for file_index, file_path in enumerate(planned, start=start_idx):
                source_id = pathlib.Path(file_path).stem
                out_tmp = os.path.join(tmp_dir, f"{source_id}.jsonl")
                if (args.resume and not args.no_resume) and (file_path in processed_prev) and os.path.exists(out_tmp):
                    to_skip.append((file_index, file_path))
                else:
                    to_process.append((file_index, file_path))

            if args.workers and args.workers > 1:
                logger.info("⚡ Parallel build enabled: workers=%s (files=%s, skip=%s)", args.workers, len(to_process), len(to_skip))
                results: List[Tuple[int, str, str, int, Dict[str, Any]]] = []

                # Main-thread aggregated session progress (workers report via queue).
                session_q: "queue.Queue[int]" = queue.Queue()
                total_sessions = sum(_count_sessions_in_file(fp) for _, fp in to_process)

                try:
                    from tqdm import tqdm  # type: ignore
                except Exception:
                    tqdm = None

                with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
                    futs = [ex.submit(_job, fi, fp) for fi, fp in to_process]
                    pending = set(futs)
                    done_boxes = 0
                    done_files = 0

                    p_files = tqdm(total=len(futs), desc="BUILD uuid files") if tqdm else None
                    p_sessions = tqdm(total=total_sessions, desc="BUILD sessions", leave=False) if (tqdm and total_sessions) else None

                    try:
                        while pending:
                            # Drain session progress reported by workers.
                            drained = 0
                            while True:
                                try:
                                    session_q.get_nowait()
                                    drained += 1
                                except queue.Empty:
                                    break
                            if drained and p_sessions:
                                p_sessions.update(drained)

                            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                            for fut in done:
                                r = fut.result()
                                results.append(r)
                                done_files += 1
                                stats = r[4] or {}
                                try:
                                    done_boxes += int(stats.get("boxes", 0) or 0)
                                except Exception:
                                    pass
                                if p_files:
                                    p_files.update(1)
                                    p_files.set_postfix({"files": f"{done_files}/{len(futs)}", "boxes": done_boxes})
                    finally:
                        # Final drain.
                        drained = 0
                        while True:
                            try:
                                session_q.get_nowait()
                                drained += 1
                            except queue.Empty:
                                break
                        if drained and p_sessions:
                            p_sessions.update(drained)
                        if p_sessions:
                            p_sessions.close()
                        if p_files:
                            p_files.close()

                # Merge in planned order: only append newly processed files this run.
                results_by_index = {r[0]: r for r in results}
                appended_boxes = 0
                for file_index, file_path in to_process:
                    _, _, out_tmp, _, _ = results_by_index[file_index]
                    appended_boxes += _append_jsonl(Config.FINAL_CONTENT_FILE, out_tmp)

                ck_obj["updated_at"] = datetime.now().isoformat()
                ck_obj["next_file_index"] = end_idx
                ck_obj["last_file"] = planned[-1] if planned else None
                processed = ck_obj.get("processed_files")
                if not isinstance(processed, list):
                    processed = []
                    ck_obj["processed_files"] = processed
                for _, fp in to_process:
                    processed.append(fp)
                ck_obj["appended_boxes_last_run"] = appended_boxes
                _save_checkpoint(Config.BUILD_CHECKPOINT_FILE, ck_obj)
            else:
                # Original sequential behavior (with checkpoint + user_id_next)
                user_id_next = int(ck_obj.get("user_id_next", 0) or 0)
                for file_index, file_path in _tqdm(list(enumerate(planned, start=start_idx)), total=len(planned), desc="BUILD"):
                    try:
                        raw_list = _load_raw_conversations(file_path)
                        # attach source id from filename for source mapping
                        source_id = pathlib.Path(file_path).stem
                        uuid8 = (source_id or "")[:8]
                        for item in raw_list:
                            if isinstance(item, dict) and "_source_id" not in item:
                                item["_source_id"] = source_id

                        # Build with numeric user_id_start=0, then rewrite to uuid-based ids.
                        if uuid8:
                            _THREAD_CTX.user_id_map = {i: (uuid8 if i == 0 else f"{uuid8}_{i}") for i in range(len(raw_list))}
                        else:
                            _THREAD_CTX.user_id_map = None
                        try:
                            boxes = builder.build_all(raw_list_override=raw_list, user_id_start=0, write_incremental=False)
                            if uuid8:
                                _apply_uuid_user_ids(boxes, uuid8=uuid8, user_id_start=0, user_count=len(raw_list))
                        finally:
                            try:
                                _THREAD_CTX.user_id_map = None
                            except Exception:
                                pass
                        builder.save_incremental(boxes, append=True)
                        user_id_next += len(raw_list)

                        ck_obj["updated_at"] = datetime.now().isoformat()
                        ck_obj["next_file_index"] = file_index + 1
                        ck_obj["last_file"] = file_path
                        ck_obj["user_id_next"] = user_id_next
                        processed = ck_obj.get("processed_files")
                        if isinstance(processed, list):
                            processed.append(file_path)
                        _save_checkpoint(Config.BUILD_CHECKPOINT_FILE, ck_obj)
                    except Exception as e:
                        ck_obj["updated_at"] = datetime.now().isoformat()
                        ck_obj["error"] = str(e)
                        ck_obj["error_file"] = file_path
                        _save_checkpoint(Config.BUILD_CHECKPOINT_FILE, ck_obj)
                        raise
        else:
            boxes = builder.build_all()
            if not Config.CHECKPOINT_EVERY_SAMPLE:
                builder.save(boxes)
        builder.summarize_and_log()
        logger.info("✅ Build done")

    if args.stage in ("retrieve", "all"):

        _announce_outputs("retrieve", [Config.SIMPLE_RETRIEVAL_JSONL, Config.SIMPLE_RETRIEVAL_CSV, Config.VECTOR_DIR])
        retr = SimpleRetriever(worker, top_k=Config.TOP_K_RETRIEVE)
        retr.run(Config.SIMPLE_RETRIEVAL_JSONL, Config.SIMPLE_RETRIEVAL_CSV)

    if args.stage in ("generate", "all"):
        _announce_outputs("generate", [Config.GENERATION_RESULT_FILE, Config.GENERATION_REPORT_CSV, Config.GEN_SUMMARY_FILE, Config.TOKEN_LOG_FILE])

        generator = AnswerGenerator(
            worker,
            answer_topn=topn_list,
            text_modes=text_modes,
            stage_label="gen",
        )
        generator.run(
            Config.SIMPLE_RETRIEVAL_JSONL,
            Config.GENERATION_RESULT_FILE,
            Config.GENERATION_REPORT_CSV,
        )


if __name__ == "__main__":
    main()
