PROMPT_MSG_CONTINUATION = """Please determine whether the current message continues with the main topic of the previous messages. Only answer Yes/No/Partially Shifted.  
previous messages: {ref}
current message: {curr}
Answer:"""

# PASS 1: Extract mentions only (no classification)
PROMPT_DIALOG_EXTRACT = """Generate a structured analysis of the provided dialog by performing the following tasks.

The dialog includes:
- A session window: "Session window: <session_start> -> <session_end>" (treat session_end as the "observation time")
- Timestamps for each message
- A "Persona info:" line, which may include the user's name, e.g. "Name: Williams Daniel; ..."

1) **Identifying salient keywords**
Extract 3-8 most salient nouns, named entities, and key terminology that represent core concepts.
Avoid generic words and prioritize specificity (e.g., "denim", "Paris", "Project X").

2) **Determining the core topic**
In one clear phrase, state the primary subject of the discussion based on the actual content.

3) **Extracting atomic memory mentions**
Identify and list ALL atomic memory mentions that are explicitly stated in the dialog and/or persona info, following these strict rules:

3.1 **Grounding (verbatim / near-verbatim)**
- Each mention MUST be directly grounded in the dialog text or the persona info line.
- Do NOT infer new facts or add details not present in the text.

3.2 **Atomicity**
- Output each atomic mention as ONE standalone natural-language sentence.
- Do NOT merge multiple facts into one mention.

3.3 **What to include**
Extract mentions for all explicitly stated items including:
- Personality & traits (including MBTI and personality tags)
- Emotional states & expressions (e.g., stress, gratitude, concern)
- Preferences WITH the WHY/reason in the SAME sentence (if the reason is present)
- Relationship / family status (e.g., married, single, no children)
- Reflections / realizations
- Past events/changes and explicit future plans/goals

3.4 **Persona info extraction**
- If a "Persona info:" line is present, extract ALL fields from persona info as separate mentions (even if not discussed in the dialog).
- Pay special attention to: personality tags, relationship status, pet preferences + reasons, and life goals with full details.

3.5 **Temporal Resolution**
- You MUST use the `<session_end>` time as the current observation time to resolve any relative temporal expressions (e.g., "yesterday", "last week", "last Thursday").
- IMPORTANT: Use the `resolve_temporal_expression` function to accurately calculate absolute dates from relative expressions. This ensures correct handling of weekdays, leap years, and month boundaries.
- After any temporal reference in your extracted mention, add the fully-resolved absolute date or date range in parentheses using the result from the function.
- Time Point Example: "...last Thursday (YYYY-MM-DD)."
- Duration Example: "...last week (YYYY-MM-DD to YYYY-MM-DD)."

3.6 **Naming & formatting rules**
- Each mention MUST be a single sentence.
- If the user's name is available (from persona info OR the dialog), use the name in the mention (e.g., "Williams Daniel..." or "Williams Daniel's...") instead of "User...".
- If no name is available, use "User...".
- Do NOT include JSON fields like start_time/end_time/type inside the mention sentence.

**Output Format**
Return a valid JSON object with the following exact keys:
{
  "keywords": ["keyword1", "keyword2",...],
  "topic": "topic phrase",
  "mentions": [
    "One atomic mention sentence",
    "Another atomic mention sentence"
  ]
}

Content to analyze:
{text}
"""

# PASS 2: Classify + assign times for each mention
PROMPT_DIALOG_CLASSIFICATION = """Generate a structured classification of the provided extracted mentions by performing the following tasks.

You will be given:
- A session window: {session_start_time} -> {session_end_time} (treat session_end as the "observation time")
- A list of extracted atomic mentions (strings)

1) **Classifying each mention**
For EACH mention, create ONE structured memory object and classify it into exactly ONE of:
- **OCCURRENCE**, **STATE**, **ATTRIBUTE**, **INTENTION**

2) **Assigning time fields**
For each object, provide:
- **description**: a single natural-language sentence derived from the mention (do NOT add new facts).
- **type**: one of OCCURRENCE / STATE / ATTRIBUTE / INTENTION
- **start_time**, **end_time**:
  - Extract the exact dates using the resolved absolute dates provided in parentheses from the mention (e.g., convert "last Thursday (2024-01-18)" to "2024-01-18").
  - Use ISO 8601 format when the mention explicitly states a resolved time/date.
  - Otherwise use the session window fallback ({session_start_time}, {session_end_time}).

3) **Category definitions**
3.1 **OCCURRENCE**
- One-time event/action or a state change (started/stopped/became/moved/graduated/got married/divorced, etc.)
- Time: use the event time if stated; otherwise fall back to the session window

3.2 **STATE**
- Ongoing/changeable status/role/relationship/preference/emotion (e.g., "is married", "likes X", "felt stressed")
- Time: if ongoing, end_time = {session_end_time}; if start unknown, start_time = {session_start_time}

3.3 **ATTRIBUTE**
- Stable/timeless identity facts or traits (name/gender/MBTI/tags/traits, birthday as a static fact)
- Time: use the session window as a placeholder

3.4 **INTENTION**
- Future-oriented plan/goal/intended action
- Time: anchor start_time/end_time to the session window; keep any future execution date only in description

4) **Rules**
- State change ⇒ OCCURRENCE
- Do not invent dates; if unknown, use session window fallback
- Do NOT include start_time/end_time/type inside the description text (only in JSON fields)

**Output Format**
Return a valid JSON object with the following exact key:
{
  "explicit_mentions": [
    {
      "description": "One sentence description derived from the mention",
      "start_time": "YYYY-MM-DD or {session_start_time}",
      "end_time": "YYYY-MM-DD or {session_end_time}",
      "type": "OCCURRENCE|STATE|ATTRIBUTE|INTENTION"
    }
  ]
}

Mentions JSON:
{mentions_json}
"""