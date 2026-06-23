PROMPT_MSG_CONTINUATION = """Please determine whether the current message continues with the main topic of the previous messages. Only answer Yes/No/Partially Shifted.  
previous messages: {ref}
current message: {curr}
Answer:"""

PROMPT_DIALOG_EXTRACT = """Generate a structured analysis of the provided dialog by performing the following tasks.

1. **Identifying salient keywords:** Extract 3-8 most salient nouns, named entities, and key terminology that represent core concepts. Avoid common words (e.g., "good", "see") and prioritize specificity.

2. **Determining the core topic:** In one clear phrase, state the primary subject or objective of the discussion based on the actual content.

3. **Extracting explicit event and plan mentions:** Identify and list only the **events, factual developments, or specific future plans** that are **explicitly mentioned** in the dialog. Follow these strict rules:
    3.1. **Focus on Verbatim or Near-Verbatim Content:** Each extracted item must be directly grounded in the dialog text. Do not infer, summarize, or combine information to create new "events."
    3.2. **Distinguish Event Types:**
        - **Past/Completed Events:** Actions or occurrences that are stated as having happened (e.g., "I went to...", "We completed the project").
        - **Established Facts/Changes:** Concrete facts or changes presented as already true (e.g., "I am now the team lead", "The system is down").
        - **Explicit Future Plans:** Specific plans for the future mentioned by the speakers (e.g., "We will meet on Friday", "I'm planning to visit Paris").
    3.3. **Exclude Non-Events:** Do NOT include:
        - General states of being (e.g., "I'm swamped", "I'm happy").
        - Questions, greetings, or expressions of intent without a plan (e.g., "We should talk sometime").
        - Vague aspirations or possibilities.
    3.4. **Temporal Resolution:** You MUST convert any relative temporal expressions (e.g., "yesterday", "last week", "tomorrow") to absolute dates based on the observation time provided above. Append the fully-resolved absolute date or date range in parentheses at the end of the extracted clause. 
        - Example: "...went to a support group yesterday (YYYY-MM-DD)."
    3.5. **Framing:** Phrase each extracted item as a concise, standalone clause that captures the core of what was mentioned.

**Output Format:** Provide the analysis as a valid JSON object with the following exact keys:
{
"keywords": ["keyword1", "keyword2", ...],
"topic": "clear topic phrase",
"explicit_mentions": [
    "One atomic mention sentence (YYYY-MM-DD)",
    "Another atomic mention sentence (YYYY-MM-DD)"
]
}
Content to analyze: {text}
"""

PROMPT_DIALOG_CLASSIFICATION = """Generate a structured classification of the provided extracted mentions by performing the following tasks.

You will be given:
- A session window: {session_start_time} -> {session_end_time} (treat session_end as the "observation time")
- A list of extracted explicit mentions (strings). These are strictly events, facts, or plans, and typically include a resolved date in parentheses at the end, e.g., "... (YYYY-MM-DD)".

1) **Classifying each mention**
For EACH mention, create ONE structured memory object and classify it into exactly ONE of:
- **OCCURRENCE**: Use for "Past/Completed Events" (e.g., went to a meeting, completed a project).
- **STATE**: Use for "Established Facts/Changes" or ongoing situations (e.g., is now the team lead, the system is down).
- **INTENTION**: Use for "Explicit Future Plans" (e.g., planning to visit Paris next Friday).
- **ATTRIBUTE**: Use ONLY if a stable, timeless identity fact was somehow extracted. Otherwise, prefer OCCURRENCE or STATE.

2) **Assigning time fields**
For each object, provide:
- **description**: A single natural-language sentence derived from the mention. **CRITICAL: You MUST remove the date parentheses (e.g., "(YYYY-MM-DD)") from this description text.** The description should be clean.
- **type**: one of OCCURRENCE / STATE / ATTRIBUTE / INTENTION
- **start_time**, **end_time**:
  - Extract the exact ISO 8601 dates from the resolved absolute dates provided in parentheses at the end of the mention.
  - If it's a single date, use it for start_time. For an ongoing STATE, set end_time to {session_end_time}.
  - If no date is found in parentheses, use the session window fallback ({session_start_time}, {session_end_time}).

**Output Format**
Return a valid JSON object with the following exact keys:
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