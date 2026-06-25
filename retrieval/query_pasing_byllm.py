import json
import re
import os
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI


# ================= 配置 =================
# ⚠️ 安全建议：不要把真实 key 写在代码里。这里保持你的默认写法不改逻辑，只提醒风险。
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://tao.plus7.plus/v1")

LLM_TIMEOUT_SEC = float(os.environ.get("LLM_TIMEOUT_SEC", "5.0"))


# ================= 1. 数据结构定义 =================

'''
用来存储从问题中提取出的“时间约束”信息。
包括开始时间（start）、结束时间（end）、时间约束的类型（是某一个时间点、一个范围、某个时间之前还是之后）、
是不是模糊时间（is_fuzzy），以及原问题中的时间短语片段（raw_text）。
'''
@dataclass
class TimeConstraint:
    start: Optional[str] = None
    end: Optional[str] = None
    type: str = "NONE"  # "POINT", "RANGE", "NONE", "ANCHOR"
    anchor_event: Optional[str] = None
    anchor_relation: Optional[str] = None
    is_fuzzy: bool = False
    raw_text: Optional[str] = None


'''
作用: 检索阶段的统一指令包。它打包了所有的解析结果，告诉下游的检索器该怎么做：
original_query: 原问题。
rewritten_query: 去掉了时间短语后的“干净”问题（专门用来做向量检索，防止时间词干扰语义）。
intent: 查询意图（如查过去的窗口期 WINDOW，查某个时间点的状态 AS_OF，查静态属性 STATIC 等）。
time_constraint: 上面提到的时间约束。
use_interval_tree: 布尔值，用于指示下游检索器是否需要启用“时间区间树”做硬性时间过滤。
parse_source: 记录这条指令是快速正则生成的（FAST）还是大模型生成的（LLM）。
'''
@dataclass
class SearchDirective:
    original_query: str
    rewritten_query: str
    intent: str
    target_types: List[str]
    time_constraint: TimeConstraint
    use_interval_tree: bool
    use_vector_store: bool
    parse_source: str = "LLM"  # "FAST" or "LLM" or "FALLBACK"
    time_axis: str = "BOTH_UNION"  # "EVENT", "SESSION", "BOTH_UNION", "NONE"


# ================= 2.5 时间轴语义检测 =================
# SESSION = "when did the user TALK to the assistant about X" (chat-history axis)
# EVENT   = "when did X happen in the real world"            (life-event axis)

_SESSION_CUE = re.compile(
    r"\b(tell|told|telling|mention(?:ed|ing)?|said|say|saying|bring\s+up|brought\s+up|"
    r"discuss(?:ed|ing)?|talk(?:ed|ing)?|chat(?:ted|ting)?|"
    r"share(?:d)?\s+with\s+you|let\s+(?:you|me)\s+know|ask(?:ed)?\s+(?:you|me))\b"
    r"|\b(?:in|during|from)\s+(?:our|the|that|last|previous|earlier|prior|recent)\s+"
    r"(?:conversation|chat|session|dialogue|talk|exchange)s?\b"
    r"|\bwe\s+(?:spoke|talked|discussed|chatted)\b"
    r"|\blast\s+time\s+(?:we|you|i)\b",
    re.I,
)

_EVENT_CUE = re.compile(
    r"\b(happen(?:ed)?|occur(?:red)?|took\s+place|went|visit(?:ed)?|meet|met|"
    r"start(?:ed)?|finish(?:ed)?|end(?:ed)?|begin|began|"
    r"graduat(?:ed|ing|e)|mov(?:ed|ing|e)|"
    r"marr(?:ied|y|ying)|divorc(?:ed|ing|e)|"
    r"work(?:ed)?|liv(?:ed|ing|e)|travel(?:ed|ing|led|ling)?|"
    r"buy|bought|sell|sold|join(?:ed)?|leave|left|retire(?:d)?|"
    r"born|die(?:d)?)\b",
    re.I,
)


def _infer_time_axis(query: str, intent: str, time_kind: str) -> str:
    """Decide which temporal axis a query refers to."""
    if time_kind in (None, "", "NONE"):
        return "NONE"
    has_session = bool(_SESSION_CUE.search(query or ""))
    has_event = bool(_EVENT_CUE.search(query or ""))
    if has_session and not has_event:
        return "SESSION"
    if has_event and not has_session:
        return "EVENT"
    return "BOTH_UNION"


def _normalize_time_axis(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    v = str(value).strip().upper()
    if v in {"EVENT", "SESSION", "BOTH_UNION", "BOTH_INTERSECT", "NONE"}:
        return v
    if v in {"BOTH", "UNION", "ANY"}:
        return "BOTH_UNION"
    if v in {"INTERSECT", "INTERSECTION", "BOTH_AND"}:
        return "BOTH_INTERSECT"
    if v in {"CHAT", "CONVERSATION", "DIALOGUE"}:
        return "SESSION"
    if v in {"REAL", "LIFE", "WORLD"}:
        return "EVENT"
    return fallback


def dispatch_temporal_filter(
    axis: Optional[str],
    query_event,
    query_session,
) -> Tuple[set, str]:
    """Apply the time-axis dispatcher on top of two callbacks.

    Returns:
        (filtered_ids, mode_used)
    """
    a = (axis or "BOTH_UNION").upper()
    if a == "NONE":
        return set(), "NONE"
    if a == "EVENT":
        return query_event(), "EVENT"
    if a == "SESSION":
        return query_session(), "SESSION"
    if a == "BOTH_INTERSECT":
        ev = query_event()
        ss = query_session()
        inter = ev & ss
        if inter:
            return inter, "BOTH_INTERSECT"
        return ev | ss, "BOTH_INTERSECT_DEGRADED_UNION"
    # Default / BOTH_UNION
    ev = query_event()
    ss = query_session()
    return ev | ss, "BOTH_UNION"


# ================= 2. Prompt =================
# 1) 明确 when/before/after/while/during + 子句 => ANCHOR（强规则）
# 2) rewritten_query 必须删除完整 time 子句
PROMPT_LITE = """Analyze the query and return JSON only. No explanations. No markdown.

Allowed intent: WINDOW | AS_OF | STATIC | PLANNING | MISC

Intent classification rules:
- WINDOW: Factual queries about events in a time period (past tense)
  * "What happened in 2025?", "What did I do last month?"
  * Key: Past tense verbs (happened, did, was)

- AS_OF: State at a specific time point (past tense)
  * "Where did I live as of Sep 05, 2025?", "Was I employed in 2024?"
  * Key: "as of", "was", "were" + specific date

- STATIC: Timeless facts and attributes
  * "What is my MBTI?", "What is my birthday?"
  * Key: Unchanging attributes, no time context

- PLANNING: Hypothetical, predictive, or future-oriented queries
  * Explicit plans: "What do I plan to do?", "What am I going to do?"
  * Hypothetical: "What could happen?", "Who might I rely on?"
  * Predictive: "What strategies could X use?", "What would X do?"
  * Key: Modal verbs (could, would, might, should) OR explicit future plans
  * IMPORTANT: Even if query mentions a specific date, if it uses modal verbs, it's PLANNING

- MISC: Unclear or mixed intent queries

Time extraction:
- time.kind: ABSOLUTE | RELATIVE_POINT | RELATIVE_RANGE | NONE
- CRITICAL distinction:
  * ABSOLUTE/RELATIVE: Specific dates or time expressions (e.g., "as of Sep 05, 2025", "after January 6, 2026", "in 2023", "last month")
    - The time phrase contains a specific date, year, month, or relative time expression
    - Use ABSOLUTE for explicit dates/years, RELATIVE_POINT for "today"/"yesterday", RELATIVE_RANGE for "last month"/"this year"

rewritten_query:
- MUST remove the entire time phrase/clause described in time.text (if any),
  and keep the remaining question.

Return JSON exactly in this schema:
{{"intent":"WINDOW|AS_OF|STATIC|PLANNING|MISC",
  "time":{"text":null,"kind":"ABSOLUTE|RELATIVE_POINT|RELATIVE_RANGE|NONE"},
  "rewritten_query":"string"}}

Examples:
Query: "Was I married when I lived in Paris?"
Return: {"intent":"AS_OF","time":{"text":"when I lived in Paris","kind":"NONE"},"rewritten_query":"Was I married?"}

Query: "What did I do after I graduated?"
Return: {{"intent":"WINDOW","time":{{"text":"after I graduated","kind":"NONE"}},"rewritten_query":"What did I do?"}}

Query: "What is my status as of Sep 05, 2025?"
Return: {{"intent":"AS_OF","time":{{"text":"as of Sep 05, 2025","kind":"ABSOLUTE"}},"rewritten_query":"What is my status?"}}

Query: "What happened after January 6, 2026?"
Return: {{"intent":"WINDOW","time":{{"text":"after January 6, 2026","kind":"ABSOLUTE"}},"rewritten_query":"What happened?"}}

Query: "What strategies could Susan use during her work in June 2036?"
Return: {{"intent":"PLANNING","time":{{"text":"in June 2036","kind":"ABSOLUTE"}},"rewritten_query":"What strategies could Susan use during her work?"}}

Query: "{query}"
"""


# ================= 3. 本地日期解析逻辑 =================
'''
作用: 本地日期推算工具箱。大模型和正则通常提取出的是相对时间文本（如 "last month"、"q1 2024"、"yesterday"），
这个类负责把这些文本“翻译”成精确的 YYYY-MM-DD 格式的绝对起止日期。
它还负责为了处理模糊时间（如 "around 2023"）而给时间范围添加缓冲期（Buffer）。
'''
class DateResolver:
    @staticmethod
    def _end_of_month(y: int, m: int) -> date:
        if m == 12:
            return date(y, 12, 31)
        return date(y, m + 1, 1) - timedelta(days=1)

    @staticmethod
    def _month_shift(d: date, delta_months: int) -> date:
        y = d.year
        m = d.month + delta_months
        while m > 12:
            y += 1
            m -= 12
        while m < 1:
            y -= 1
            m += 12
        day = min(d.day, DateResolver._end_of_month(y, m).day)
        return date(y, m, day)

    @staticmethod
    def _week_range(d: date) -> Tuple[date, date]:
        monday = d - timedelta(days=d.weekday())
        sunday = monday + timedelta(days=6)
        return monday, sunday

    @staticmethod
    def _quarter_range(year: int, q: int) -> Tuple[date, date]:
        start_month = 1 + (q - 1) * 3
        s = date(year, start_month, 1)
        e = DateResolver._end_of_month(year, start_month + 2)
        return s, e

    @staticmethod
    def resolve(text: Optional[str], kind: str, base_time: datetime) -> Tuple[Optional[str], Optional[str], bool]:
        if not text or kind == "NONE":
            return None, None, False

        t = text.strip().lower()
        is_fuzzy = any(w in t for w in ["around", "about", "approx", "roughly", "circa", "ca."])

        t_clean = re.sub(r'\b(around|about|approx\.?|roughly|circa|ca\.?)\b', '', t).strip()
        t_clean = re.sub(r'\b(in|on|at)\b\s+', '', t_clean).strip()

        today = base_time.date()

        if kind == "RELATIVE_POINT":
            if "yesterday" in t_clean:
                d = today - timedelta(days=1)
                s = d.isoformat()
                return s, s, is_fuzzy
            if "today" in t_clean:
                s = today.isoformat()
                return s, s, is_fuzzy
            if "tomorrow" in t_clean:
                d = today + timedelta(days=1)
                s = d.isoformat()
                return s, s, is_fuzzy
            return None, None, is_fuzzy

        if kind == "RELATIVE_RANGE":
            if "week" in t_clean:
                base = today
                if "last week" in t_clean:
                    base = today - timedelta(days=7)
                elif "next week" in t_clean:
                    base = today + timedelta(days=7)
                s, e = DateResolver._week_range(base)
                return s.isoformat(), e.isoformat(), is_fuzzy

            if "month" in t_clean:
                base = today.replace(day=15)
                if "last month" in t_clean:
                    base = DateResolver._month_shift(base, -1)
                elif "next month" in t_clean:
                    base = DateResolver._month_shift(base, +1)
                y, m = base.year, base.month
                s = date(y, m, 1)
                e = DateResolver._end_of_month(y, m)
                return s.isoformat(), e.isoformat(), is_fuzzy

            if "year" in t_clean:
                y = today.year
                if "last year" in t_clean:
                    y -= 1
                elif "next year" in t_clean:
                    y += 1
                return f"{y}-01-01", f"{y}-12-31", is_fuzzy

            return None, None, is_fuzzy

        m = re.fullmatch(r"q([1-4])\s*(\d{4})", t_clean)
        if m:
            q = int(m.group(1))
            y = int(m.group(2))
            s, e = DateResolver._quarter_range(y, q)
            return s.isoformat(), e.isoformat(), is_fuzzy

        m = re.fullmatch(r"(early|mid|late)\s+(\d{4})", t_clean)
        if m:
            part = m.group(1)
            y = int(m.group(2))
            if part == "early":
                s, e = date(y, 1, 1), date(y, 4, 30)
            elif part == "mid":
                s, e = date(y, 5, 1), date(y, 8, 31)
            else:
                s, e = date(y, 9, 1), date(y, 12, 31)
            return s.isoformat(), e.isoformat(), is_fuzzy

        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t_clean)
        if m:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            s = d.isoformat()
            return s, s, is_fuzzy

        # 支持自然语言日期格式: "Sep 05, 2025", "January 6, 2026", "as of Sep 05, 2025", "after January 6, 2026"
        # 先移除前缀词
        t_date = re.sub(r'\b(as of|after|before|on|at|in)\b\s*', '', t_clean).strip()

        # Check year-only pattern after removing prefixes
        m = re.fullmatch(r"(\d{4})", t_date)
        if m:
            y = int(m.group(1))
            return f"{y}-01-01", f"{y}-12-31", is_fuzzy

        try:
            from dateutil import parser as date_parser
            parsed_date = date_parser.parse(t_date, fuzzy=True)
            s = parsed_date.date().isoformat()
            return s, s, is_fuzzy
        except:
            pass

        return None, None, is_fuzzy

    @staticmethod
    def apply_buffer(start: str, end: str, is_fuzzy: bool) -> Tuple[str, str]:
        if not start or not end:
            return start, end

        s_dt = datetime.strptime(start, "%Y-%m-%d")
        e_dt = datetime.strptime(end, "%Y-%m-%d")
        span = (e_dt - s_dt).days

        days_to_add = 0
        if is_fuzzy:
            if span > 300:
                days_to_add = 60
            elif span > 25:
                days_to_add = 10
            else:
                days_to_add = 3

        if days_to_add > 0:
            s_dt -= timedelta(days=days_to_add)
            e_dt += timedelta(days=days_to_add)

        return s_dt.strftime("%Y-%m-%d"), e_dt.strftime("%Y-%m-%d")


# ================= 4. 快速解析器（修 3） =================
# 修复点：
# 3) anchor 检测从简单字符串 => 正则扩大覆盖：when/before/after/while/during 等出现就交给 LLM
'''
作用: 正则快速提取器。
它内置了一系列关键词（INTENT_KEYWORDS）和正则表达式（PATTERNS）。
它的目的是先对问题进行一次快速扫描，如果问题非常简单标准（比如明确包含 "in 2023" 或 "what is"），
它就能瞬间推断出意图和时间，直接绕过调用大模型，从而极大节省等待延迟和 API 成本费用。
'''
class FastParser:
    PATTERNS = [
        # "by" patterns (must come first to capture "by" as BEFORE)
        (re.compile(r'\b(by)\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+(19|20)\d{2}\b', re.I), "ABSOLUTE"),
        (re.compile(r'\b(by)\s+(19|20)\d{2}\b', re.I), "ABSOLUTE"),

        # After/Before/Since + full date
        (re.compile(r'\b(after|before|since)\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+(19|20)\d{2}\b', re.I), "ABSOLUTE"),

        # Full date first: Month Day, Year
        (re.compile(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+(19|20)\d{2}\b', re.I), "ABSOLUTE"),

        # ISO date
        (re.compile(r'\b(19|20)\d{2}-\d{2}-\d{2}\b', re.I), "ABSOLUTE"),

        # Month + Year (e.g., June 2036)
        (re.compile(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(19|20)\d{2}\b', re.I), "ABSOLUTE"),

        # After/Before/Since + year
        (re.compile(r'\b(after|before|since)\s+(19|20)\d{2}\b', re.I), "ABSOLUTE"),

        # Relative time points
        (re.compile(r'\b(yesterday|today|tomorrow)\b', re.I), "RELATIVE_POINT"),

        # Relative time ranges
        (re.compile(r'\b(last year|this year|next year|last month|this month|next month|last week|this week|next week)\b', re.I), "RELATIVE_RANGE"),

        # Quarter / period
        (re.compile(r'\bq[1-4]\s*(19|20)\d{2}\b', re.I), "ABSOLUTE"),
        (re.compile(r'\b(early|mid|late)\s+(19|20)\d{2}\b', re.I), "ABSOLUTE"),

        # Year last: avoid eating year inside full date first
        (re.compile(r'\b((?:around|about|approx\.?|roughly|circa|ca\.)\s+)?(19|20)\d{2}\b', re.I), "ABSOLUTE"),
    ]

    INTENT_KEYWORDS = {
        # Order matters: check more specific patterns first
        "STATIC": [
            "what is", "what's",  # Must come before AS_OF "is "
            "birth date", "birthday", "my name", "blood type", "mbti",
            "middle name"
        ],
        "AS_OF": [
            "as of",  # Strong signal
            "where did i live", "was i", "who was", "where was",
            " unemployed as of", " employed as of"  # More specific
        ],
        "WINDOW": [
            "what happened", "what did i do", "when did i",
            "did i go", "did i visit", "did i express",
            "did i work", "did i dislike", "did i like", 
            "therapy session","during the dialogue"
        ],
        "PLANNING": [
            # Explicit future plans
            "plan to", "going to", "intend to", "future goal",
            " explore next", " influence ", " prefer if",
            # Modal verbs indicating hypothetical/predictive queries
            "could ", "would ", "might ", "should ",
            # Hypothetical question patterns
            "what could", "what would", "what might", "what should",
            "who could", "who would", "who might", "who should",
            "how could", "how would", "how might", "how should",
            "which could", "which would", "which might",
            # Predictive/recommendation patterns
            "what strategies could", "what strategies might",
            "who might ", "what might "
        ]
    }

    _REWRITE_TIME_NOISE = re.compile(
        r'\b(around|about|approx\.?|roughly|circa|ca\.|in|on|at|last|this|next|year|month|week|'
        r'yesterday|today|tomorrow|q[1-4]|as of|after|before|by)\b',
        re.I
    )

    # Anchor hint: semantic anchors like "when I lived in Paris", "during his career", "while working"
    # Expanded to include possessive pronouns (his/her/my/our/their) and articles (the/a/an)
    _ANCHOR_HINT = re.compile(
        r'\b(when|while|during)\s+(?:'
        r'(?:i|we|he|she|they)\b|'  # Personal pronouns
        r'(?:his|her|my|our|their|your)\b|'  # Possessive pronouns
        r'(?:the|a|an)\s+\w+(?:\s+\w+){0,3}|'  # Articles followed by 1-4 words (e.g., "the financial crisis")
        r'\w+ing\b'  # Verb+ing forms (e.g., "working", "living", "studying")
        r')',
        re.I
    )
    # Explicit date whitelist:
    # 1) YYYY-MM-DD
    # 2) Month Day, Year  (e.g., Apr 25, 2027)
    # 3) standalone year  (e.g., 2027)
    _EXPLICIT_DATE_HINT = re.compile(
        r'\b('
        r'(?:19|20)\d{2}-\d{2}-\d{2}'
        r'|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+(?:19|20)\d{2}'
        r'|(?:19|20)\d{2}'
        r')\b',
        re.I
    )

    @staticmethod
    def try_parse(query: str) -> Optional[Dict[str, Any]]:
        q_lower = query.lower()

        intent = "MISC"
        # Check intents in order (STATIC first for "what is" precedence)
        for k, v in FastParser.INTENT_KEYWORDS.items():
            if any(phrase in q_lower for phrase in v):
                intent = k
                break

        # Anchor interception:
        # - keep LLM fallback for semantic anchor queries
        # - but if explicit date exists, allow Fast path to continue
        has_anchor = bool(FastParser._ANCHOR_HINT.search(query))
        has_explicit_date = bool(FastParser._EXPLICIT_DATE_HINT.search(query))
        if has_anchor and not has_explicit_date:
            return None

        time_text = None
        time_kind = "NONE"
        for pat, kind in FastParser.PATTERNS:
            m = pat.search(query)
            if m:
                time_text = m.group(0).strip()
                time_kind = kind
                break

        # Conservative mode: prefer LLM for ambiguous cases
        # Use fast path only when we have BOTH intent AND time, OR very clear intent
        if intent == "MISC" and time_kind == "NONE":
            return None

        # If we have time but no clear intent, use fast path (time is reliable)
        # If we have intent but no time, allow STATIC and PLANNING to stay on fast path:
        # - STATIC: original behavior
        # - PLANNING: usually ends up use_interval_tree=False, so LLM is often unnecessary latency
        if time_kind == "NONE" and intent not in ["STATIC", "PLANNING"]:
            return None

        rewritten = query
        rewritten = FastParser._REWRITE_TIME_NOISE.sub("", rewritten)
        # Remove month name + day + year patterns
        rewritten = re.sub(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+(19|20)\d{2}\b', '', rewritten, flags=re.I)
        # Remove YYYY-MM-DD date patterns (complete dates only, not partial)
        rewritten = re.sub(r'\b(19|20)\d{2}-\d{2}-\d{2}\b', "", rewritten)
        # Note: Removed standalone year removal to avoid creating incomplete dates like "-07-20"
        # Time info is already captured in time_constraint, so rewritten_query doesn't need dates
        rewritten = re.sub(r'\s+', ' ', rewritten).strip(" ?.,")
        if not rewritten:
            rewritten = query

        return {
            "intent": intent,
            "time": {"text": time_text, "kind": time_kind},
            "rewritten_query": rewritten
        }


# ================= 5. 主解析器（修 4、5） =================
# 修复点：
# 4) intent 兜底：如果 query 强命中 AS_OF 特征（如 "Was I married ..."）且 LLM 给 MISC，则提升到 AS_OF


class QueryParser:
    # Semantic anchors: expanded to include possessive pronouns, articles, and verb+ing forms
    # Examples: "when I lived", "during his career", "while the crisis", "while working"
    _ANCHOR_HINT = re.compile(
        r'\b(when|while|during)\s+(?:'
        r'(?:i|we|he|she|they)\b|'  # Personal pronouns
        r'(?:his|her|my|our|their|your)\b|'  # Possessive pronouns
        r'(?:the|a|an)\s+\w+(?:\s+\w+){0,3}|'  # Articles followed by 1-4 words
        r'\w+ing\b'  # Verb+ing forms (e.g., "working", "living")
        r')',
        re.I
    )
    _ASOF_STRONG = re.compile(r"\bwas i\b", re.I)
    _STATUS_WORDS = re.compile(r"\b(married|single|divorced|engaged)\b", re.I)

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.model = model

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=LLM_TIMEOUT_SEC)
        self._cache: Dict[str, SearchDirective] = {}
        self.use_fast_path: bool = True

    def _cache_key(self, query: str, day: date) -> str:
        return f"{day.isoformat()}::{query.strip().lower()}"

    @staticmethod
    def _extract_anchor_clause(original_query: str) -> Optional[str]:
        """
        从首次出现 when/before/after/while/during 开始，尽量提取到句尾。
        例:
          "Was I married when I lived in Paris?" => "when I lived in Paris"
          "What did I do after I graduated?" => "after I graduated"
        """
        m = re.search(r"\b(when|before|after|while|during)\b.*", original_query, re.I)
        if not m:
            return None
        clause = m.group(0).strip()
        clause = clause.strip(" ?.,;:")  # 去掉尾部标点
        return clause if clause else None

    @staticmethod
    def _remove_time_clause(query: str, clause: str) -> str:
        """
        从 query 中删除 time clause（大小写不敏感），并清理多余空格/标点。
        """
        if not clause:
            return query
        pattern = re.compile(re.escape(clause), re.I)
        out = pattern.sub("", query)
        out = re.sub(r"\s+", " ", out).strip()
        out = out.strip(" ?.,;:")
        if not out:
            return query
        return out

    def parse(self, query: str, base_time: Optional[datetime] = None) -> SearchDirective:
        q = (query or "").strip()
        if not q:
            return self._fallback_directive(query or "")

        ref_day = (base_time.date() if base_time else datetime.now().date())
        ck = self._cache_key(q, ref_day)
        if ck in self._cache:
            return self._cache[ck]

        # Stage A: Fast
        fast_result = FastParser.try_parse(q)
        if fast_result:
            fast_intent = (fast_result.get("intent") or "MISC").upper()
            fast_time = fast_result.get("time") or {}
            fast_kind = (fast_time.get("kind") or "NONE").upper()

            if fast_intent != "MISC" or fast_kind != "NONE":
                d = self._construct_directive(q, fast_result, source="FAST", base_time=base_time)
                self._cache[ck] = d
                return d

        # Stage B: LLM
        raw_content = None
        try:
            prompt = PROMPT_LITE.replace("{query}", q)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=256
            )
            raw_content = (response.choices[0].message.content or "").strip()

            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```json|^```|```$", "", raw_content).strip()

            data = json.loads(raw_content)
            d = self._construct_directive(q, data, source="LLM", base_time=base_time)
            self._cache[ck] = d
            return d
        except Exception as e:
            print(f"LLM Parsing failed for query: '{q}'\nError: {e}")
            if raw_content is not None:
                print(f"Raw LLM Output: {raw_content}")

            if fast_result:
                d = self._construct_directive(q, fast_result, source="FAST", base_time=base_time)
            else:
                d = self._fallback_directive(q)

            self._cache[ck] = d
        return d

    def _construct_directive(self, original_query: str, data: Dict[str, Any], source: str,base_time: Optional[datetime] = None) -> SearchDirective:
        intent = (data.get("intent") or "MISC").upper()

        time_info = data.get("time") or {}
        time_text = time_info.get("text")
        time_kind = (time_info.get("kind") or "NONE").upper()

        if time_kind not in {"ABSOLUTE", "RELATIVE_POINT", "RELATIVE_RANGE", "NONE"}:
            time_kind = "NONE"

        extracted_clause = None
        if self._ANCHOR_HINT.search(original_query) and (time_kind == "NONE" and not time_text):
            clause = self._extract_anchor_clause(original_query)
            if clause:
                extracted_clause = clause
                time_text = clause
                time_kind = "NONE"

        if intent == "MISC":
            if self._ASOF_STRONG.search(original_query) and self._STATUS_WORDS.search(original_query):
                intent = "AS_OF"

        type_map = {
            "WINDOW": ["OCCURRENCE"],
            "AS_OF": ["STATE"],
            "STATIC": ["ATTRIBUTE"],
            "PLANNING": ["INTENTION"]
        }
        target_types = type_map.get(intent, ["OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"])

        ref_now = base_time if base_time is not None else datetime.now()
        start, end, is_fuzzy_text = DateResolver.resolve(time_text, time_kind, ref_now)

        is_fuzzy_intent = bool(is_fuzzy_text)

        if start and end:
            start, end = DateResolver.apply_buffer(start, end, is_fuzzy_intent)

        anchor_event = None
        anchor_relation = None
        time_type_final = "NONE"

        if start and end:
            t = (time_text or "").lower()
            if "after" in t or "since" in t:
                time_type_final = "AFTER"
            elif "before" in t or "until" in t or "by" in t:
                time_type_final = "BEFORE"
            else:
                time_type_final = "RANGE" if start != end else "POINT"

        use_tree = False
        use_vector = True

        if intent in {"PLANNING", "STATIC"}:
            use_tree = False
        else:
            if time_type_final in {"RANGE", "POINT", "AFTER", "BEFORE"}:
                use_tree = True

        rewritten = data.get("rewritten_query", original_query)

        if time_text:
            if re.search(re.escape(time_text), rewritten, re.I):
                rewritten = self._remove_time_clause(rewritten, time_text)
            elif rewritten.strip() == original_query.strip():
                rewritten = self._remove_time_clause(rewritten, time_text)

        # Infer time axis for bi-temporal filtering
        time_axis = _infer_time_axis(original_query, intent, time_type_final)
        if not use_tree:
            time_axis = "NONE"

        return SearchDirective(
            original_query=original_query,
            rewritten_query=rewritten,
            intent=intent,
            target_types=target_types,
            time_constraint=TimeConstraint(
                start=start,
                end=end,
                type=time_type_final,
                anchor_event=anchor_event,
                anchor_relation=anchor_relation,
                is_fuzzy=is_fuzzy_intent,
                raw_text=time_text
            ),
            use_interval_tree=use_tree,
            use_vector_store=use_vector,
            parse_source=source,
            time_axis=time_axis,
        )

    def _fallback_directive(self, query: str) -> SearchDirective:
        return SearchDirective(
            query,
            query,
            "MISC",
            ["OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"],
            TimeConstraint(type="NONE"),
            False,
            True,
            "FALLBACK",
            "NONE",
        )



if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, required=True, help="Query string")
    args = parser.parse_args()

    try:
        query_parser = QueryParser()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n📝 Parsing: \"{args.query}\"")
    start = time.time()
    d = query_parser.parse(args.query)
    lat = time.time() - start

    print(f"\n🔍 Result (Source: {d.parse_source}, Latency: {lat:.4f}s)")
    print(json.dumps(d, default=lambda o: o.__dict__, indent=2, ensure_ascii=False))
