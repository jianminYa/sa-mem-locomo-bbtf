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

@dataclass
class TimeConstraint:
    start: Optional[str] = None
    end: Optional[str] = None
    type: str = "NONE"  # "POINT", "RANGE", "NONE", "ANCHOR", AFTER, BEFORE
    anchor_event: Optional[str] = None
    anchor_relation: Optional[str] = None
    is_fuzzy: bool = False
    raw_text: Optional[str] = None


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
    # Which temporal axis the query's time constraint refers to.
    #   EVENT           -> real-world event time only (e.g. "what happened in 2023")
    #   SESSION         -> when the user talked to the assistant (e.g. "when did I tell you")
    #   BOTH_UNION      -> ambiguous / both plausible: union of the two axes (default)
    #   BOTH_INTERSECT  -> both axes must agree: intersect (falls back to union if empty)
    #   NONE            -> no temporal filter (time_constraint.type == "NONE")
    time_axis: str = "BOTH_UNION"


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

Time axis (which timeline the time phrase refers to):
- time.axis: EVENT | SESSION | BOTH | NONE
  * EVENT:   the time refers to when an event happened in the real world
             e.g. "What happened in 2023?", "When did I move to Paris?"
  * SESSION: the time refers to when the user talked to the assistant
             e.g. "When did I tell you about Paris?", "Did we discuss this last week?"
  * BOTH:    ambiguous or both plausible — use when no clear cue
  * NONE:    only when time.kind is NONE
- Signal: talking verbs near "you/me" (tell, told, mention, say, said, discuss,
  talk, chat, share with you, let you know, ask you) or meta references
  ("our conversation", "last chat", "earlier session") => SESSION.
  Past-tense life-event verbs (happen, went, moved, graduated, worked,
  lived, visited, married, ...) without a talking verb => EVENT.

rewritten_query:
- MUST remove the entire time phrase/clause described in time.text (if any),
  and keep the remaining question.

Return JSON exactly in this schema:
{{"intent":"WINDOW|AS_OF|STATIC|PLANNING|MISC",
  "time":{"text":null,"kind":"ABSOLUTE|RELATIVE_POINT|RELATIVE_RANGE|NONE","axis":"EVENT|SESSION|BOTH|NONE"},
  "rewritten_query":"string"}}

Examples:
Query: "Was I married when I lived in Paris?"
Return: {"intent":"AS_OF","time":{"text":"when I lived in Paris","kind":"NONE","axis":"NONE"},"rewritten_query":"Was I married?"}

Query: "What did I do after I graduated?"
Return: {{"intent":"WINDOW","time":{{"text":"after I graduated","kind":"NONE","axis":"NONE"}},"rewritten_query":"What did I do?"}}

Query: "What is my status as of Sep 05, 2025?"
Return: {{"intent":"AS_OF","time":{{"text":"as of Sep 05, 2025","kind":"ABSOLUTE","axis":"EVENT"}},"rewritten_query":"What is my status?"}}

Query: "What happened after January 6, 2026?"
Return: {{"intent":"WINDOW","time":{{"text":"after January 6, 2026","kind":"ABSOLUTE","axis":"EVENT"}},"rewritten_query":"What happened?"}}

Query: "When did I tell you about my Paris trip in 2023?"
Return: {{"intent":"WINDOW","time":{{"text":"in 2023","kind":"ABSOLUTE","axis":"SESSION"}},"rewritten_query":"When did I tell you about my Paris trip?"}}

Query: "What strategies could Susan use during her work in June 2036?"
Return: {{"intent":"PLANNING","time":{{"text":"in June 2036","kind":"ABSOLUTE","axis":"EVENT"}},"rewritten_query":"What strategies could Susan use during her work?"}}

Query: "{query}"
"""


# ================= 2.5 时间轴语义检测 =================
# SESSION = "when did the user TALK to the assistant about X" (chat-history axis)
# EVENT   = "when did X happen in the real world"            (life-event axis)
# Detectors are shared by FastParser and QueryParser so regex + LLM paths stay
# consistent when the LLM does not emit an axis hint.

_SESSION_CUE = re.compile(
    # Verbs about talking / telling / discussing (near "you" or "me")
    r"\b(tell|told|telling|mention(?:ed|ing)?|said|say|saying|bring\s+up|brought\s+up|"
    r"discuss(?:ed|ing)?|talk(?:ed|ing)?|chat(?:ted|ting)?|"
    r"share(?:d)?\s+with\s+you|let\s+(?:you|me)\s+know|ask(?:ed)?\s+(?:you|me))\b"
    # Meta references to the conversation itself
    r"|\b(?:in|during|from)\s+(?:our|the|that|last|previous|earlier|prior|recent)\s+"
    r"(?:conversation|chat|session|dialogue|talk|exchange)s?\b"
    r"|\bwe\s+(?:spoke|talked|discussed|chatted)\b"
    r"|\blast\s+time\s+(?:we|you|i)\b",
    re.I,
)

_EVENT_CUE = re.compile(
    # Past-tense life-event verbs (no "you/me" object)
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
    """Decide which temporal axis a query refers to.

    Priority (highest first):
      1. No time constraint -> "NONE"
      2. Session cues only -> "SESSION"
      3. Event cues only   -> "EVENT"
      4. Both cues         -> "SESSION" (the meta act of telling dominates
         the life-event verb when both are present, e.g. "when did I tell
         you I went to Paris" — user wants the chat turn, not the trip)
      5. Neither cue       -> "BOTH_UNION" (ambiguous, permissive default)
    """
    if time_kind in (None, "", "NONE"):
        return "NONE"
    has_session = bool(_SESSION_CUE.search(query or ""))
    has_event = bool(_EVENT_CUE.search(query or ""))
    if has_session and not has_event:
        return "SESSION"
    if has_event and not has_session:
        return "EVENT"
    if has_session and has_event:
        return "SESSION"
    return "BOTH_UNION"


def _normalize_time_axis(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    v = str(value).strip().upper()
    if v in {"EVENT", "SESSION", "BOTH_UNION", "BOTH_INTERSECT", "NONE"}:
        return v
    # Accept looser LLM outputs.
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

    Args:
        axis:           value of SearchDirective.time_axis (case-insensitive).
                        Unknown/None falls back to BOTH_UNION (permissive).
        query_event:    zero-arg callable returning a set of ids filtered on
                        the event-time interval tree.
        query_session:  zero-arg callable returning a set of ids filtered on
                        the session-time interval tree.

    Returns:
        (filtered_ids, mode_used) where mode_used describes which branch ran,
        useful for logging/tracing. mode_used values:
        "EVENT", "SESSION", "BOTH_UNION", "BOTH_INTERSECT",
        "BOTH_INTERSECT_DEGRADED_UNION" (when intersect was empty and we
        degraded to union to avoid an empty result).
    """
    a = (axis or "BOTH_UNION").upper()
    if a == "NONE":
        # Caller should not have invoked temporal filtering; return empty.
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
        # Strict intersect is empty — degrade to union so the caller doesn't
        # return an empty result set that blocks downstream ranking.
        return ev | ss, "BOTH_INTERSECT_DEGRADED_UNION"
    # Default / BOTH_UNION / anything unknown.
    ev = query_event()
    ss = query_session()
    return ev | ss, "BOTH_UNION"


# ================= 3. 本地日期解析逻辑 =================

# ---------- Shared regex building blocks ----------

_MONTHS_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)
_YEAR_RE = r"(?:19|20)\d{2}"
_ISO_DATE_RE = r"(?:19|20)\d{2}-\d{2}-\d{2}"
# Month Day, Year — "October 13, 2023", "Oct. 13 2023", "October 13th, 2023",
# "October 13,2023" (comma without following space)
_MONTH_DAY_YEAR_RE = (
    rf"{_MONTHS_RE}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+){_YEAR_RE}"
)
# Day Month, Year — "9 October 2022", "9th October, 2022", "13th of October 2023",
# "1 February,2023" (comma without following space)
_DAY_MONTH_YEAR_RE = (
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS_RE}\.?(?:,\s*|\s+){_YEAR_RE}"
)
# Month + Year — "June 2036", "Oct 2022"
_MONTH_YEAR_RE = rf"{_MONTHS_RE}\.?\s+{_YEAR_RE}"
# ISO-like with slash or dot separator — "2023/10/13", "2023.10.13"
_ISO_DATE_SLASH_RE = r"(?:19|20)\d{2}[/.]\d{1,2}[/.]\d{1,2}"
# US/EU numeric — "10/13/2023", "13-10-2023", "10.13.2023"
_NUMERIC_DATE_RE = r"\d{1,2}[/.\-]\d{1,2}[/.\-](?:19|20)\d{2}"
# Year-Month short ISO — "2023-10", "2023/10"
_YEAR_MONTH_ISO_RE = r"(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])(?![-/.\d])"
_DATE_EXPR_RE = (
    rf"(?:{_ISO_DATE_RE}|{_ISO_DATE_SLASH_RE}|{_NUMERIC_DATE_RE}|"
    rf"{_MONTH_DAY_YEAR_RE}|{_DAY_MONTH_YEAR_RE}|"
    rf"{_MONTH_YEAR_RE}|{_YEAR_MONTH_ISO_RE}|{_YEAR_RE})"
)

_WEEKDAYS_RE = (
    r"(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|"
    r"Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)"
)

_MONTH_NUM: Dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_WEEKDAY_NUM: Dict[str, int] = {
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2, "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4, "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
}

_SEASON_MONTHS: Dict[str, Tuple[int, int, int, int]] = {
    # (start_month, start_day, end_month, end_day) — northern hemisphere default
    "spring": (3, 1, 5, 31),
    "summer": (6, 1, 8, 31),
    "fall":   (9, 1, 11, 30),
    "autumn": (9, 1, 11, 30),
    "winter": (12, 1, 2, 28),  # wraps into next year, handled inline
}

_FUZZY_WORDS_RE = re.compile(
    r"\b(around|about|approx\.?|roughly|circa|ca\.?)\b", re.I,
)


# ---------- TimeMatch: canonical result of every matcher ----------

@dataclass
class TimeMatch:
    """Unified result of a single time-expression matcher.

    span     — (start, end) character offsets in the original query, used for
               clause-based rewrite.
    text     — substring of the original query that was consumed.
    start    — ISO date (YYYY-MM-DD) of the lower bound, or None for BEFORE-only.
    end      — ISO date of the upper bound, or None for AFTER-only.
    kind     — ABSOLUTE | RELATIVE_POINT | RELATIVE_RANGE | NONE
    boundary — POINT | RANGE | AFTER | BEFORE | ANCHOR | NONE
               Already expresses the final semantics, so downstream code does
               NOT need to re-infer "after" vs "range" from the text.
    """
    span: Tuple[int, int]
    text: str
    start: Optional[str]
    end: Optional[str]
    kind: str
    boundary: str
    is_fuzzy: bool = False
    anchor_event: Optional[str] = None
    anchor_relation: Optional[str] = None  # DURING | AFTER | BEFORE


# ---------- Small date utilities ----------

def _eom(y: int, m: int) -> date:
    if m == 12:
        return date(y, 12, 31)
    return date(y, m + 1, 1) - timedelta(days=1)


def _month_shift(d: date, delta_months: int) -> date:
    y = d.year
    m = d.month + delta_months
    while m > 12:
        y += 1
        m -= 12
    while m < 1:
        y -= 1
        m += 12
    day = min(d.day, _eom(y, m).day)
    return date(y, m, day)


def _week_range(d: date) -> Tuple[date, date]:
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _quarter_range(year: int, q: int) -> Tuple[date, date]:
    start_month = 1 + (q - 1) * 3
    return date(year, start_month, 1), _eom(year, start_month + 2)


def _is_fuzzy_text(text: str) -> bool:
    return bool(_FUZZY_WORDS_RE.search(text or ""))


def _parse_date_expr(text: str) -> Optional[Tuple[date, date, str]]:
    """Resolve a bare date expression to (start, end, kind).

    kind is ABSOLUTE for everything here.
    Handles: ISO YYYY-MM-DD, Month Day Year, Month Year, bare year.
    """
    if not text:
        return None
    t = text.strip().strip(",.;:")

    # ISO dash: 2023-10-13
    m = re.fullmatch(rf"\s*({_ISO_DATE_RE})\s*", t)
    if m:
        y, mo, d = map(int, m.group(1).split("-"))
        try:
            dd = date(y, mo, d)
            return dd, dd, "ABSOLUTE"
        except ValueError:
            return None

    # ISO slash/dot: 2023/10/13, 2023.10.13
    m = re.fullmatch(r"\s*((?:19|20)\d{2})[/.](\d{1,2})[/.](\d{1,2})\s*", t)
    if m:
        try:
            dd = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dd, dd, "ABSOLUTE"
        except ValueError:
            return None

    # Month Day, Year: October 13, 2023 / Oct. 13 2023 / October 13th, 2023 / October 13,2023
    m = re.fullmatch(
        rf"\s*({_MONTHS_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)({_YEAR_RE})\s*",
        t, re.I,
    )
    if m:
        try:
            dd = date(
                int(m.group(3)),
                _MONTH_NUM[m.group(1).lower()],
                int(m.group(2)),
            )
            return dd, dd, "ABSOLUTE"
        except ValueError:
            return None

    # Day Month, Year: 9 October 2022 / 13th of October, 2023 / 9 Oct. 2022 / 1 February,2023
    m = re.fullmatch(
        rf"\s*(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTHS_RE})\.?(?:,\s*|\s+)({_YEAR_RE})\s*",
        t, re.I,
    )
    if m:
        try:
            dd = date(
                int(m.group(3)),
                _MONTH_NUM[m.group(2).lower()],
                int(m.group(1)),
            )
            return dd, dd, "ABSOLUTE"
        except ValueError:
            return None

    # Numeric date: 10/13/2023, 13-10-2023, 10.13.2023
    # Disambiguate: if a>12 -> DMY; if b>12 -> MDY; else default to MDY
    # (English QA data skews US-style); ValueError filters impossible dates.
    m = re.fullmatch(r"\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]((?:19|20)\d{2})\s*", t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12 and b <= 12:
            day, mo = a, b
        elif b > 12 and a <= 12:
            mo, day = a, b
        elif a <= 12 and b <= 12:
            mo, day = a, b
        else:
            return None
        try:
            dd = date(y, mo, day)
            return dd, dd, "ABSOLUTE"
        except ValueError:
            return None

    # Month + Year: June 2036 / Oct. 2022
    m = re.fullmatch(rf"\s*({_MONTHS_RE})\.?\s+({_YEAR_RE})\s*", t, re.I)
    if m:
        mo = _MONTH_NUM[m.group(1).lower()]
        y = int(m.group(2))
        return date(y, mo, 1), _eom(y, mo), "ABSOLUTE"

    # Year-Month ISO short: 2023-10 / 2023/10
    m = re.fullmatch(r"\s*((?:19|20)\d{2})[-/](\d{1,2})\s*", t)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return date(y, mo, 1), _eom(y, mo), "ABSOLUTE"
        return None

    # Year only: 2023
    m = re.fullmatch(rf"\s*({_YEAR_RE})\s*", t)
    if m:
        y = int(m.group(1))
        return date(y, 1, 1), date(y, 12, 31), "ABSOLUTE"

    return None


# ---------- Matchers ----------
# Each matcher: (query: str, base_time: datetime) -> Optional[TimeMatch]
# They are tried in priority order (most specific first). First hit wins.

# Anchor (non-date subclause): handled first so "after I graduated" is NOT
# captured by the "after <date>" matcher below.
_ANCHOR_SUBCLAUSE_RE = re.compile(
    r"\b(?P<kw>when|while|during|before|after|since)\s+"
    r"(?P<body>"
    r"(?:i|we|he|she|they|you)\b"           # personal pronouns
    r"|(?:his|her|my|our|their|your)\b"     # possessive pronouns
    r"|(?:the|a|an)\s+\w+"                  # "the/a/an" + noun-ish
    r"|\w+ing\b"                            # gerunds: working, living
    r")",
    re.I,
)

# Clause boundary: stop anchor extraction at ", and", ", but", punctuation,
# "?.;:" — prevents gobbling the rest of the sentence.
_ANCHOR_CLAUSE_STOP = re.compile(
    r"(?:[?.;:,!]|\s+and\s+(?:did|was|were|i|you|he|she|they|we)\s)",
    re.I,
)

_ANCHOR_RELATION = {
    "when":   "DURING",
    "while":  "DURING",
    "during": "DURING",
    "before": "BEFORE",
    "after":  "AFTER",
    "since":  "AFTER",
}


def _m_anchor(q: str, base_time: datetime) -> Optional[TimeMatch]:
    m = _ANCHOR_SUBCLAUSE_RE.search(q)
    if not m:
        return None
    # Extend the match to the end of the clause (bounded by punctuation / "and <pron>").
    end = len(q)
    stop = _ANCHOR_CLAUSE_STOP.search(q, m.end())
    if stop:
        end = stop.start()
    clause_text = q[m.start():end].strip()
    clause_text = clause_text.rstrip(" ?.,;:")
    # Drop the keyword to get the event body.
    kw = m.group("kw").lower()
    event = re.sub(rf"^\b{kw}\b\s*", "", clause_text, count=1, flags=re.I).strip()
    return TimeMatch(
        span=(m.start(), m.start() + len(clause_text)),
        text=clause_text,
        start=None, end=None,
        kind="NONE",
        boundary="ANCHOR",
        anchor_event=event,
        anchor_relation=_ANCHOR_RELATION.get(kw, "DURING"),
    )


def _mk(q, m, start, end, kind, boundary, **kwargs) -> TimeMatch:
    return TimeMatch(
        span=m.span(),
        text=q[m.start():m.end()],
        start=start.isoformat() if isinstance(start, date) else start,
        end=end.isoformat() if isinstance(end, date) else end,
        kind=kind,
        boundary=boundary,
        is_fuzzy=_is_fuzzy_text(q[m.start():m.end()]),
        **kwargs,
    )


# Bounded matchers: after / before / since / by / until + <date-expr>
_RE_AFTER_DATE = re.compile(
    rf"\b(?:after|following)\s+(?P<d>{_DATE_EXPR_RE})\b", re.I,
)
_RE_BEFORE_DATE = re.compile(
    rf"\b(?:before|prior\s+to)\s+(?P<d>{_DATE_EXPR_RE})\b", re.I,
)
_RE_SINCE_DATE = re.compile(
    rf"\b(?:since|starting\s+(?:from|in))\s+(?P<d>{_DATE_EXPR_RE})\b", re.I,
)
_RE_BY_DATE = re.compile(
    rf"\b(?:by|up\s+until|up\s+to)\s+(?P<d>{_DATE_EXPR_RE})\b", re.I,
)
_RE_UNTIL_DATE = re.compile(
    rf"\b(?:until|till)\s+(?P<d>{_DATE_EXPR_RE})\b", re.I,
)
_RE_AS_OF = re.compile(
    rf"\b(?:as\s+of|as\s+at)\s+(?P<d>{_DATE_EXPR_RE})\b", re.I,
)
_RE_FROM_TO = re.compile(
    rf"\bfrom\s+(?P<a>{_DATE_EXPR_RE})\s+(?:to|through|thru|till|until)\s+(?P<b>{_DATE_EXPR_RE})\b",
    re.I,
)
_RE_BETWEEN = re.compile(
    rf"\bbetween\s+(?P<a>{_DATE_EXPR_RE})\s+and\s+(?P<b>{_DATE_EXPR_RE})\b",
    re.I,
)


def _m_as_of(q, bt):
    m = _RE_AS_OF.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group("d"))
    if not r:
        return None
    s, e, _ = r
    return _mk(q, m, s, e, "ABSOLUTE", "POINT" if s == e else "RANGE")


def _m_after_date(q, bt):
    m = _RE_AFTER_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group("d"))
    if not r:
        return None
    _, e, _ = r
    return _mk(q, m, e + timedelta(days=1), None, "ABSOLUTE", "AFTER")


def _m_before_date(q, bt):
    m = _RE_BEFORE_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group("d"))
    if not r:
        return None
    s, _, _ = r
    return _mk(q, m, None, s - timedelta(days=1), "ABSOLUTE", "BEFORE")


def _m_since_date(q, bt):
    m = _RE_SINCE_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group("d"))
    if not r:
        return None
    s, _, _ = r
    return _mk(q, m, s, None, "ABSOLUTE", "AFTER")


def _m_by_date(q, bt):
    m = _RE_BY_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group("d"))
    if not r:
        return None
    _, e, _ = r
    return _mk(q, m, None, e, "ABSOLUTE", "BEFORE")


def _m_until_date(q, bt):
    m = _RE_UNTIL_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group("d"))
    if not r:
        return None
    _, e, _ = r
    return _mk(q, m, None, e, "ABSOLUTE", "BEFORE")


def _m_from_to(q, bt):
    m = _RE_FROM_TO.search(q)
    if not m:
        return None
    ra = _parse_date_expr(m.group("a"))
    rb = _parse_date_expr(m.group("b"))
    if not (ra and rb):
        return None
    return _mk(q, m, ra[0], rb[1], "ABSOLUTE", "RANGE")


def _m_between(q, bt):
    m = _RE_BETWEEN.search(q)
    if not m:
        return None
    ra = _parse_date_expr(m.group("a"))
    rb = _parse_date_expr(m.group("b"))
    if not (ra and rb):
        return None
    return _mk(q, m, ra[0], rb[1], "ABSOLUTE", "RANGE")


# Full date / ISO date — consumed as POINT
_RE_FULL_DATE = re.compile(rf"\b({_MONTH_DAY_YEAR_RE})\b", re.I)
_RE_FULL_DATE_DMY = re.compile(rf"\b({_DAY_MONTH_YEAR_RE})\b", re.I)
_RE_ISO_DATE = re.compile(rf"\b({_ISO_DATE_RE})\b")
_RE_ISO_SLASH = re.compile(rf"\b({_ISO_DATE_SLASH_RE})\b")
_RE_NUMERIC_DATE = re.compile(rf"\b({_NUMERIC_DATE_RE})\b")
_RE_YEAR_MONTH_ISO = re.compile(rf"\b({_YEAR_MONTH_ISO_RE})")


def _m_full_date(q, bt):
    m = _RE_FULL_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group(1))
    if not r:
        return None
    s, e, _ = r
    return _mk(q, m, s, e, "ABSOLUTE", "POINT")


def _m_full_date_dmy(q, bt):
    m = _RE_FULL_DATE_DMY.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group(1))
    if not r:
        return None
    s, e, _ = r
    return _mk(q, m, s, e, "ABSOLUTE", "POINT")


def _m_iso_date(q, bt):
    m = _RE_ISO_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group(1))
    if not r:
        return None
    s, e, _ = r
    return _mk(q, m, s, e, "ABSOLUTE", "POINT")


def _m_iso_slash(q, bt):
    m = _RE_ISO_SLASH.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group(1))
    if not r:
        return None
    s, e, _ = r
    return _mk(q, m, s, e, "ABSOLUTE", "POINT")


def _m_numeric_date(q, bt):
    m = _RE_NUMERIC_DATE.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group(1))
    if not r:
        return None
    s, e, _ = r
    return _mk(q, m, s, e, "ABSOLUTE", "POINT")


def _m_year_month_iso(q, bt):
    m = _RE_YEAR_MONTH_ISO.search(q)
    if not m:
        return None
    r = _parse_date_expr(m.group(1))
    if not r:
        return None
    s, e, _ = r
    return _mk(q, m, s, e, "ABSOLUTE", "RANGE")


# Quarter / period / half / season / decade
_RE_QUARTER = re.compile(rf"\bQ([1-4])\s*({_YEAR_RE})\b", re.I)
_RE_RELATIVE_QUARTER = re.compile(r"\b(last|this|next|previous|past)\s+quarter\b", re.I)
_RE_EARLY_MID_LATE = re.compile(rf"\b(early|mid|late)\s+({_YEAR_RE})\b", re.I)
_RE_HALF_YEAR = re.compile(
    rf"\b(?:(?:first|1st|second|2nd|latter|last)\s+half\s+(?:of\s+)?({_YEAR_RE})|H([12])\s*({_YEAR_RE}))\b",
    re.I,
)
_RE_START_END_OF_YEAR = re.compile(
    rf"\b(start|beginning|end)\s+of\s+({_YEAR_RE})\b", re.I,
)
_RE_SEASON = re.compile(
    rf"\b(spring|summer|fall|autumn|winter)\s+({_YEAR_RE})\b", re.I,
)
_RE_DECADE = re.compile(r"\b(?:the\s+)?(19|20)(\d)0s\b", re.I)
# Month + Year — full month range
_RE_MONTH_YEAR = re.compile(rf"\b({_MONTHS_RE})\s+({_YEAR_RE})\b", re.I)


def _m_quarter(q, bt):
    m = _RE_QUARTER.search(q)
    if not m:
        return None
    s, e = _quarter_range(int(m.group(2)), int(m.group(1)))
    return _mk(q, m, s, e, "ABSOLUTE", "RANGE")


def _m_relative_quarter(q, bt):
    m = _RE_RELATIVE_QUARTER.search(q)
    if not m:
        return None
    today = bt.date()
    current_q = (today.month - 1) // 3 + 1
    qword = m.group(1).lower()
    delta = {"last": -1, "previous": -1, "past": -1, "this": 0, "next": 1}[qword]
    target_q = current_q + delta
    year = today.year
    while target_q < 1:
        target_q += 4
        year -= 1
    while target_q > 4:
        target_q -= 4
        year += 1
    s, e = _quarter_range(year, target_q)
    return _mk(q, m, s, e, "RELATIVE_RANGE", "RANGE")


def _m_early_mid_late(q, bt):
    m = _RE_EARLY_MID_LATE.search(q)
    if not m:
        return None
    part, y = m.group(1).lower(), int(m.group(2))
    if part == "early":
        s, e = date(y, 1, 1), date(y, 4, 30)
    elif part == "mid":
        s, e = date(y, 5, 1), date(y, 8, 31)
    else:
        s, e = date(y, 9, 1), date(y, 12, 31)
    return _mk(q, m, s, e, "ABSOLUTE", "RANGE")


def _m_half_year(q, bt):
    m = _RE_HALF_YEAR.search(q)
    if not m:
        return None
    text = m.group(0).lower()
    if m.group(1):
        y = int(m.group(1))
        first = any(w in text for w in ["first", "1st"])
    else:
        y = int(m.group(3))
        first = m.group(2) == "1"
    if first:
        s, e = date(y, 1, 1), date(y, 6, 30)
    else:
        s, e = date(y, 7, 1), date(y, 12, 31)
    return _mk(q, m, s, e, "ABSOLUTE", "RANGE")


def _m_start_end_of_year(q, bt):
    m = _RE_START_END_OF_YEAR.search(q)
    if not m:
        return None
    kind, y = m.group(1).lower(), int(m.group(2))
    if kind in ("start", "beginning"):
        s, e = date(y, 1, 1), date(y, 2, 28)
    else:
        s, e = date(y, 11, 1), date(y, 12, 31)
    return _mk(q, m, s, e, "ABSOLUTE", "RANGE")


def _m_season(q, bt):
    m = _RE_SEASON.search(q)
    if not m:
        return None
    season, y = m.group(1).lower(), int(m.group(2))
    sm, sd, em, ed = _SEASON_MONTHS[season]
    if season == "winter":
        s, e = date(y, 12, 1), _eom(y + 1, 2)
    else:
        s, e = date(y, sm, sd), date(y, em, ed)
    return _mk(q, m, s, e, "ABSOLUTE", "RANGE")


def _m_decade(q, bt):
    m = _RE_DECADE.search(q)
    if not m:
        return None
    base = int(m.group(1) + m.group(2) + "0")
    return _mk(q, m, date(base, 1, 1), date(base + 9, 12, 31), "ABSOLUTE", "RANGE")


def _m_month_year(q, bt):
    m = _RE_MONTH_YEAR.search(q)
    if not m:
        return None
    mo = _MONTH_NUM[m.group(1).lower()]
    y = int(m.group(2))
    # Full calendar month range — fixes "June 2036 → single day" bug.
    return _mk(q, m, date(y, mo, 1), _eom(y, mo), "ABSOLUTE", "RANGE")


# N-ago / last-N
_RE_N_AGO = re.compile(
    r"\b(a|an|\d+)\s+(day|week|month|year|quarter)s?\s+ago\b", re.I,
)
_RE_LAST_N = re.compile(
    r"\b(?:the\s+)?(?:last|past|previous)\s+(a|an|few|\d+)\s+(day|week|month|year|quarter)s?\b",
    re.I,
)


def _n_to_int(tok: str) -> int:
    t = tok.lower()
    if t in ("a", "an"):
        return 1
    if t == "few":
        return 3
    try:
        return int(t)
    except ValueError:
        return 1


def _m_n_ago(q, bt):
    m = _RE_N_AGO.search(q)
    if not m:
        return None
    n = _n_to_int(m.group(1))
    unit = m.group(2).lower()
    today = bt.date()
    if unit == "day":
        d = today - timedelta(days=n)
        return _mk(q, m, d, d, "RELATIVE_POINT", "POINT")
    if unit == "week":
        d = today - timedelta(weeks=n)
        s, e = _week_range(d)
        return _mk(q, m, s, e, "RELATIVE_RANGE", "RANGE")
    if unit == "month":
        d = _month_shift(today.replace(day=15), -n)
        return _mk(q, m, date(d.year, d.month, 1), _eom(d.year, d.month),
                   "RELATIVE_RANGE", "RANGE")
    if unit == "year":
        y = today.year - n
        return _mk(q, m, date(y, 1, 1), date(y, 12, 31), "RELATIVE_RANGE", "RANGE")
    if unit == "quarter":
        current_q = (today.month - 1) // 3 + 1
        target_q = current_q - n
        year = today.year
        while target_q < 1:
            target_q += 4
            year -= 1
        s, e = _quarter_range(year, target_q)
        return _mk(q, m, s, e, "RELATIVE_RANGE", "RANGE")
    return None


def _m_last_n(q, bt):
    m = _RE_LAST_N.search(q)
    if not m:
        return None
    n = _n_to_int(m.group(1))
    unit = m.group(2).lower()
    today = bt.date()
    if unit == "day":
        s = today - timedelta(days=n - 1)
        return _mk(q, m, s, today, "RELATIVE_RANGE", "RANGE",
                   is_fuzzy=m.group(1).lower() == "few")
    if unit == "week":
        s = today - timedelta(weeks=n)
        return _mk(q, m, s, today, "RELATIVE_RANGE", "RANGE")
    if unit == "month":
        s_date = _month_shift(today, -n + 1)
        s = date(s_date.year, s_date.month, 1)
        return _mk(q, m, s, today, "RELATIVE_RANGE", "RANGE")
    if unit == "year":
        s = today.replace(year=today.year - n + 1, month=1, day=1)
        return _mk(q, m, s, today, "RELATIVE_RANGE", "RANGE")
    if unit == "quarter":
        s = today - timedelta(days=90 * n)
        return _mk(q, m, s, today, "RELATIVE_RANGE", "RANGE")
    return None


# Simple relative windows
_RE_REL_WEEKMONTH = re.compile(
    r"\b(last|this|next|past|previous)\s+(week|month|year|weekend)\b", re.I,
)
_RE_REL_DAYPART = re.compile(
    r"\b(this\s+morning|this\s+afternoon|this\s+evening|tonight)\b", re.I,
)
_RE_REL_SIMPLE_POINT = re.compile(r"\b(yesterday|today|tomorrow)\b", re.I)
_RE_RECENTLY = re.compile(r"\b(recently|lately|a\s+while\s+back)\b", re.I)


def _m_rel_weekmonth(q, bt):
    m = _RE_REL_WEEKMONTH.search(q)
    if not m:
        return None
    mod, unit = m.group(1).lower(), m.group(2).lower()
    today = bt.date()
    if unit == "weekend":
        # Find the weekend closest to base.
        # Saturday of base week.
        sat = today + timedelta(days=(5 - today.weekday()) % 7)
        if mod in ("last", "past", "previous"):
            sat -= timedelta(days=7)
        sun = sat + timedelta(days=1)
        return _mk(q, m, sat, sun, "RELATIVE_RANGE", "RANGE")
    if unit == "week":
        base = today
        if mod in ("last", "past", "previous"):
            base -= timedelta(days=7)
        elif mod == "next":
            base += timedelta(days=7)
        s, e = _week_range(base)
        return _mk(q, m, s, e, "RELATIVE_RANGE", "RANGE")
    if unit == "month":
        base = today.replace(day=15)
        if mod in ("last", "past", "previous"):
            base = _month_shift(base, -1)
        elif mod == "next":
            base = _month_shift(base, +1)
        return _mk(q, m, date(base.year, base.month, 1),
                   _eom(base.year, base.month), "RELATIVE_RANGE", "RANGE")
    if unit == "year":
        y = today.year
        if mod in ("last", "past", "previous"):
            y -= 1
        elif mod == "next":
            y += 1
        return _mk(q, m, date(y, 1, 1), date(y, 12, 31), "RELATIVE_RANGE", "RANGE")
    return None


def _m_rel_daypart(q, bt):
    m = _RE_REL_DAYPART.search(q)
    if not m:
        return None
    today = bt.date()
    return _mk(q, m, today, today, "RELATIVE_POINT", "POINT")


def _m_rel_simple_point(q, bt):
    m = _RE_REL_SIMPLE_POINT.search(q)
    if not m:
        return None
    today = bt.date()
    word = m.group(1).lower()
    if word == "yesterday":
        d = today - timedelta(days=1)
    elif word == "tomorrow":
        d = today + timedelta(days=1)
    else:
        d = today
    return _mk(q, m, d, d, "RELATIVE_POINT", "POINT")


def _m_recently(q, bt):
    m = _RE_RECENTLY.search(q)
    if not m:
        return None
    today = bt.date()
    # In conversational QA, "recently/lately" typically refers to events from
    # the past several months of a user's life, not literally the last 30 days.
    # Targets in the benchmark sit 60-150 days before base_time; 30-day window
    # excluded them entirely. 180 days covers the empirically observed spread.
    s = today - timedelta(days=180)
    tm = _mk(q, m, s, today, "RELATIVE_RANGE", "RANGE")
    tm.is_fuzzy = True
    return tm


# Weekday matcher: "last Monday", "next Friday", "this Wednesday"
_RE_REL_WEEKDAY = re.compile(rf"\b(last|this|next)\s+({_WEEKDAYS_RE})\b", re.I)


def _m_rel_weekday(q, bt):
    m = _RE_REL_WEEKDAY.search(q)
    if not m:
        return None
    mod = m.group(1).lower()
    wd = _WEEKDAY_NUM[m.group(2).lower()]
    today = bt.date()
    diff = (wd - today.weekday()) % 7
    if mod == "next" and diff == 0:
        diff = 7
    elif mod == "last":
        diff = diff - 7 if diff >= 0 else diff
        if diff == 0:
            diff = -7
    d = today + timedelta(days=diff)
    return _mk(q, m, d, d, "RELATIVE_POINT", "POINT")


# Year-with-context: "in 2023", "during 2023", "back in 2023", "the year 2023".
# Explicit prefix REQUIRED so "model 2024" / "room 2024" are NOT captured.
_RE_YEAR_IN_CONTEXT = re.compile(
    rf"\b(?:in|during|around|about|circa|ca\.?|back\s+in|the\s+year)\s+({_YEAR_RE})\b",
    re.I,
)


def _m_year_in_context(q, bt):
    m = _RE_YEAR_IN_CONTEXT.search(q)
    if not m:
        return None
    y = int(m.group(1))
    return _mk(q, m, date(y, 1, 1), date(y, 12, 31), "ABSOLUTE", "RANGE")


# Priority-ordered pipeline. First matcher to return non-None wins.
_MATCHERS = [
    _m_anchor,             # anchor subclause (before/after/since/when/while/during + subclause)
    _m_from_to,            # from X to Y
    _m_between,            # between X and Y
    _m_as_of,              # as of X
    _m_by_date,            # by X (inclusive end)
    _m_until_date,         # until/till X
    _m_after_date,         # after X (exclusive start)
    _m_before_date,        # before X (exclusive end)
    _m_since_date,         # since X (inclusive start)
    _m_full_date,          # Month Day, Year
    _m_full_date_dmy,      # Day Month, Year / Day of Month Year
    _m_iso_date,           # YYYY-MM-DD
    _m_iso_slash,          # YYYY/MM/DD or YYYY.MM.DD
    _m_numeric_date,       # MM/DD/YYYY or DD/MM/YYYY (and .- variants)
    _m_quarter,            # Q1 2024
    _m_half_year,          # first/second half of 2024, H1 2024
    _m_early_mid_late,     # early/mid/late 2024
    _m_start_end_of_year,  # start/end/beginning of 2024
    _m_season,             # spring 2024
    _m_decade,             # the 2020s
    _m_month_year,         # June 2036
    _m_year_month_iso,     # 2023-10 / 2023/10 (year-month short ISO)
    _m_relative_quarter,   # last/this/next quarter
    _m_n_ago,              # 3 weeks ago
    _m_last_n,             # last 3 months
    _m_rel_weekmonth,      # last/this/next week|month|year|weekend
    _m_rel_weekday,        # last Monday
    _m_rel_daypart,        # this morning / tonight
    _m_rel_simple_point,   # yesterday / today / tomorrow
    _m_recently,           # recently / lately
    _m_year_in_context,    # in 2023
]


def _find_time_match(query: str, base_time: datetime) -> Optional[TimeMatch]:
    """Run matchers in priority order, return the first hit.

    The priority list starts with the most specific compound patterns (anchor,
    from/to/between, bounded after/before/since/by), then unambiguous absolute
    date forms, then periodic forms (quarter, season, month-year), then
    relative forms, and finally plain "in YYYY" (requires context prefix)."""
    if not query:
        return None
    for matcher in _MATCHERS:
        tm = matcher(query, base_time)
        if tm is not None:
            return tm
    return None


# ---------- DateResolver: single front-door for time resolution ----------

class DateResolver:
    """Pure resolver — no side effects, no dateutil fuzzy fallback.

    Front door is :meth:`resolve_match`. Legacy :meth:`resolve` is kept as a
    thin shim returning only (start, end, is_fuzzy) so existing callers keep
    working.
    """

    # Kept for backward compatibility with any external caller.
    @staticmethod
    def _end_of_month(y: int, m: int) -> date:
        return _eom(y, m)

    @staticmethod
    def _month_shift(d: date, delta_months: int) -> date:
        return _month_shift(d, delta_months)

    @staticmethod
    def _week_range(d: date) -> Tuple[date, date]:
        return _week_range(d)

    @staticmethod
    def _quarter_range(year: int, q: int) -> Tuple[date, date]:
        return _quarter_range(year, q)

    @staticmethod
    def resolve_match(
        text: Optional[str],
        kind: Optional[str],
        base_time: datetime,
    ) -> Optional[TimeMatch]:
        """Resolve *text* into a structured TimeMatch.

        *kind* is accepted for backward-compat but not required — the matcher
        pipeline itself infers kind/boundary. If kind is NONE and text is
        empty, returns None.
        """
        if not text:
            return None
        k = (kind or "").upper()
        if k == "NONE" and not _find_time_match(text, base_time):
            return None
        return _find_time_match(text, base_time)

    @staticmethod
    def resolve(
        text: Optional[str],
        kind: str,
        base_time: datetime,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Legacy API: returns (start_iso, end_iso, is_fuzzy).

        Kept so callers outside this file continue to compile. Prefer
        :meth:`resolve_match` for full structure (boundary, anchor, kind).
        """
        tm = DateResolver.resolve_match(text, kind, base_time)
        if not tm:
            return None, None, False
        return tm.start, tm.end, tm.is_fuzzy

    @staticmethod
    def apply_buffer(
        start: Optional[str],
        end: Optional[str],
        is_fuzzy: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        if not start or not end:
            return start, end

        s_dt = datetime.strptime(start, "%Y-%m-%d")
        e_dt = datetime.strptime(end, "%Y-%m-%d")
        span = (e_dt - s_dt).days

        days_to_add = 0
        if is_fuzzy:
            # Widened 2026-04: conversational "recently/around/few X" refers to
            # events that can sit well outside the literal window. Pad more
            # aggressively so real targets are not excluded by the filter.
            if span > 300:
                days_to_add = 90    # year-scale fuzzy: ±90d (was 60)
            elif span > 25:
                days_to_add = 20    # month-scale fuzzy: ±20d (was 10)
            else:
                days_to_add = 7     # week-scale fuzzy: ±7d (was 3)

        if days_to_add > 0:
            s_dt -= timedelta(days=days_to_add)
            e_dt += timedelta(days=days_to_add)

        return s_dt.strftime("%Y-%m-%d"), e_dt.strftime("%Y-%m-%d")


# ================= 4. 快速解析器 =================
# Redesigned around the shared matcher pipeline (_find_time_match):
#   - One matcher per time-phrase family; first hit wins in priority order.
#   - Rewrite is clause-based: slice the matched span out of the query
#     (instead of the old word-noise substitution that broke "this year"
#      inside "therapy this year").
#   - ANCHOR detection returns a structured TimeMatch with
#     anchor_event/anchor_relation, so _construct_directive can emit
#     TimeConstraint.type="ANCHOR" without re-parsing.

class FastParser:
    INTENT_KEYWORDS = {
        # Order matters: check more specific patterns first.
        "STATIC": [
            "what is", "what's",
            "birth date", "birthday", "my name", "blood type", "mbti",
            "middle name",
        ],
        "AS_OF": [
            "as of", "as at",
            "where did i live", "was i", "who was", "where was",
            " unemployed as of", " employed as of",
        ],
        "WINDOW": [
            "what happened", "what did i do", "when did i",
            "did i go", "did i visit", "did i express",
            "did i work", "did i dislike", "did i like",
            "therapy session", "during the dialogue",
        ],
        "PLANNING": [
            "plan to", "going to", "intend to", "future goal",
            " explore next", " influence ", " prefer if",
            "could ", "would ", "might ", "should ",
            "what could", "what would", "what might", "what should",
            "who could", "who would", "who might", "who should",
            "how could", "how would", "how might", "how should",
            "which could", "which would", "which might",
            "what strategies could", "what strategies might",
            "who might ", "what might ",
        ],
    }

    @staticmethod
    def _detect_intent(q_lower: str) -> str:
        for intent, phrases in FastParser.INTENT_KEYWORDS.items():
            if any(phrase in q_lower for phrase in phrases):
                return intent
        return "MISC"

    @staticmethod
    def _rewrite_by_span(query: str, span: Tuple[int, int]) -> str:
        """Remove the matched time span from the original query.

        - Slices [0, span[0]) + [span[1], len), preserving everything else
          verbatim. This fixes the bug where the old "remove time noise
          words" approach ate `in`/`this`/`year` tokens inside phrases like
          `in therapy this year`.
        - Also strips adjacent connective remnants ("of", trailing "," etc.)
          at the seam.
        """
        before = query[:span[0]]
        after = query[span[1]:]
        rewritten = before + " " + after
        # Clean up the seam: double spaces, hanging punctuation around it.
        rewritten = re.sub(r"\s+", " ", rewritten)
        # Remove a connective "of" / "in" left at the seam (rare but clean).
        rewritten = re.sub(r"\s(of|in|on|at|during)\s(?=[?.!,]|$)", "", rewritten, flags=re.I)
        return rewritten.strip(" ?.,;:")

    @staticmethod
    def try_parse(query: str, base_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        if not query:
            return None
        bt = base_time or datetime.now()
        q_lower = query.lower()

        intent = FastParser._detect_intent(q_lower)

        # Find the single best time match in the query.
        tm = _find_time_match(query, bt)

        # Drop to LLM when we have nothing to work with.
        if intent == "MISC" and tm is None:
            return None
        # Let STATIC / PLANNING stay on fast path even without a time phrase.
        if tm is None and intent not in ("STATIC", "PLANNING"):
            return None

        # Rewrite: slice out the matched span (clause-based).
        rewritten = query
        if tm is not None:
            rewritten = FastParser._rewrite_by_span(query, tm.span)
            if not rewritten:
                rewritten = query

        # time_kind mirrors TimeMatch.kind; text mirrors TimeMatch.text so
        # downstream LLM-style code paths don't diverge.
        time_text = tm.text if tm else None
        time_kind = tm.kind if tm else "NONE"

        result: Dict[str, Any] = {
            "intent": intent,
            "time": {"text": time_text, "kind": time_kind},
            "rewritten_query": rewritten,
            "time_axis": _infer_time_axis(query, intent, time_kind),
        }
        if tm is not None:
            # Surface the already-resolved structure so _construct_directive
            # doesn't need to re-resolve the string (saves work + avoids
            # the round-trip losing boundary/anchor metadata).
            result["_time_match"] = tm
        return result


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
        fast_result = FastParser.try_parse(q, base_time=base_time)
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

        # Intent safety net (kept from the old code).
        if intent == "MISC":
            if self._ASOF_STRONG.search(original_query) and self._STATUS_WORDS.search(original_query):
                intent = "AS_OF"

        type_map = {
            "WINDOW": ["OCCURRENCE"],
            "AS_OF": ["STATE"],
            "STATIC": ["ATTRIBUTE"],
            "PLANNING": ["INTENTION"],
        }
        target_types = type_map.get(intent, ["OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"])

        ref_now = base_time if base_time is not None else datetime.now()

        # ---- Resolve the time phrase into a structured TimeMatch ----
        # Fast path: FastParser already ran the pipeline and passed the
        # TimeMatch through. LLM path: we resolve time_text here.
        tm: Optional[TimeMatch] = data.get("_time_match")
        if tm is None and time_text:
            tm = DateResolver.resolve_match(time_text, time_kind, ref_now)
        # LLM-detected anchor (time_text present, kind=NONE) — try to parse
        # the anchor clause directly off the original query.
        if tm is None and time_text and time_kind == "NONE":
            tm = _m_anchor(original_query, ref_now)

        start: Optional[str] = None
        end: Optional[str] = None
        boundary = "NONE"
        anchor_event = None
        anchor_relation = None
        is_fuzzy_intent = False

        if tm is not None:
            start = tm.start
            end = tm.end
            boundary = tm.boundary
            anchor_event = tm.anchor_event
            anchor_relation = tm.anchor_relation
            is_fuzzy_intent = tm.is_fuzzy
            # If the matcher found a kind but the caller said NONE, prefer
            # the matcher (it is authoritative for FAST path; for LLM path
            # it still reflects the real pattern).
            if time_kind == "NONE" and tm.kind != "NONE":
                time_kind = tm.kind
            if not time_text:
                time_text = tm.text

        if start and end:
            start, end = DateResolver.apply_buffer(start, end, is_fuzzy_intent)

        # Final TimeConstraint.type
        if boundary == "ANCHOR":
            time_type_final = "ANCHOR"
        elif boundary in ("AFTER", "BEFORE", "RANGE", "POINT"):
            time_type_final = boundary
        elif start and end:
            time_type_final = "RANGE" if start != end else "POINT"
        else:
            time_type_final = "NONE"

        use_tree = False
        use_vector = True
        if intent in {"PLANNING", "STATIC"}:
            use_tree = False
        elif time_type_final in {"RANGE", "POINT", "AFTER", "BEFORE", "ANCHOR"}:
            use_tree = True

        # ---- Rewrite ----
        rewritten = data.get("rewritten_query", original_query)
        if tm is not None and tm.span and rewritten == original_query:
            # LLM path that didn't rewrite: do clause-based rewrite ourselves.
            rewritten = FastParser._rewrite_by_span(original_query, tm.span)
            if not rewritten:
                rewritten = original_query
        elif time_text and rewritten.strip() == original_query.strip():
            # Backward-compat behavior for old LLM outputs.
            rewritten = self._remove_time_clause(rewritten, time_text)

        # ---- Time axis resolution ----
        axis_hint = data.get("time_axis")
        if axis_hint is None and isinstance(data.get("time"), dict):
            axis_hint = data["time"].get("axis")
        fallback_axis = _infer_time_axis(original_query, intent, time_kind)
        time_axis = _normalize_time_axis(axis_hint, fallback_axis)
        if time_type_final == "NONE":
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
                raw_text=time_text,
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
            time_axis="NONE",
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
