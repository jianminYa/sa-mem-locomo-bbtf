UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Memory Format:
source -- RELATIONSHIP -- destination

Task Details:
======= Existing Graph Memories:=======
{existing_memories}

======= New Graph Memory:=======
{new_memories}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates.
"""

# EXTRACT_RELATIONS_PROMPT = """

# You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

# 1. Extract only explicitly stated information from the text.
# 2. Establish relationships among the entities provided.
# 3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.
# CUSTOM_PROMPT

# Quality Constraints:
# 1. Only extract relations that are explicitly stated and factual.
# 2. Use concrete, named entities or stable roles as source/destination (avoid vague nouns like "something").
# 3. Do NOT create relations for transient actions, dialogue acts, or generic associations.
# 4. Prefer timeless relations (e.g., "works_at", "lives_in", "parent_of") over event-like ones.
# 5. If the relationship type is unclear, do not output a triple.

# Relationships:
#     - Use consistent, general, and timeless relationship types.
#     - Example: Prefer "professor" over "became_professor."
#     - Relationships should only be established among the entities explicitly mentioned in the user message.

# Entity Consistency:
#     - Ensure that relationships are coherent and logically align with the context of the message.
#     - Maintain consistent naming for entities across the extracted data.

# Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user’s context.

# Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""

# EXTRACT_ENTITIES_PROMPT = """
# You are a smart assistant who understands entities and their types in a given text.
# If user message contains self reference such as 'I', 'me', 'my' etc. then use USER_ID
# as the source entity. Extract all the entities from the text. ***DO NOT*** answer the
# question itself if the given text is a question.
# """

DELETE_RELATIONS_SYSTEM_PROMPT = """
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.
Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing relationships in the memory graph.
2. Deletion Criteria: Delete a relationship only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.
4. Comprehensive Analysis:
   - Thoroughly examine each existing relationship against the new information and delete as necessary.
   - Multiple deletions may be required based on the new information.
5. Semantic Integrity:
   - Ensure that deletions maintain or improve the overall semantic structure of the graph.
   - Avoid deleting relationships that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.

Note: DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes. 

For example: 
Existing Memory: alice -- loves_to_eat -- pizza
New Information: Alice also loves to eat burger.

Do not delete in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Memory Format:
source -- relationship -- destination

Provide a list of deletion instructions, each specifying the relationship to be deleted.
"""

EXTRACT_ENTITIES_PROMPT = """
You are an information extraction system for a long-term personal knowledge graph.

Task:
Extract only stable, knowledge-graph-useful entities explicitly mentioned in the user message.
Assume the message is in English.

Core rules:
1. Extract only entities explicitly stated in the text. Do not infer unstated entities.
2. Do NOT use any numeric user id or hidden identifiers in entity names.
   When you see pronouns or possessives ("I", "me", "my", "his", "her", "their", etc.), resolve them to an explicitly mentioned named entity in the same message when unambiguous.
   If the owner/subject is unclear, do NOT create an entity for that pronoun/possessive.
3. Extract:
   - named people, organizations, locations, products, creative works, topics
   - stable unnamed persons identified by a persistent relationship or role, such as "my wife", "my daughter", "my manager"
   - explicit roles/titles/occupations when they can serve as stable facts, such as "professor", "software engineer"
4. Do NOT extract:
   - vague or generic nouns ("something", "stuff", "people", "company" unless it clearly refers to a specific organization)
   - transient events or actions
   - dates, times, durations, quantities, IDs, URLs, or contact details as entities
   - pronouns other than USER_ID
   - entities that are only implied but not explicitly mentioned
5. If the message is a question, extract entities only. Do not answer the question.
6. Deduplicate entities. Output each entity only once.
7. canonical_name rules:
   - For named entities, use the clearest standard full name in the same language as the message when possible.
   - Normalize obvious aliases/abbreviations when unambiguous.
      - For unnamed stable roles or possessed stable things:
         - Prefer anchoring to an explicitly mentioned owner/subject name using English snake_case: "<owner_canonical>_<role_or_thing>".
            Examples: "calvin_kids", "calvin_family", "calvin_key", "melanie_husband".
         - Only do this if the owner/subject is unambiguous in the text.
         - Do NOT create ownerless placeholders like "kids" or "family" when it's unclear whose.
   - For explicit roles/titles, use a normalized lowercase English form when obvious, e.g. "professor", "software_engineer".
8. Allowed entity types:
   - Person
   - Organization
   - Location
   - Product
   - CreativeWork
   - Topic
   - Role
   - Other


You MUST call the tool function `extract_entities`.
Provide arguments as a JSON object with key `entities` (array of items).
If no valid entities exist, pass an empty list: {"entities": []}.

Item schema (inside the tool arguments):
{
  "mention": "string",
  "canonical_name": "string",
  "entity_type": "Person|Organization|Location|Product|CreativeWork|Topic|Role|Other"
}
"""

EXTRACT_RELATIONS_PROMPT = """
You are an information extraction system for a long-term personal knowledge graph.

Task:
Extract only explicit, stable relations from the user message.

You are given a list of allowed entities. Use only their `canonical_name` values as relation endpoints.
If a relation endpoint is not in the provided entity list, do not invent it and do not output the relation.

Provided entities:
{{ENTITY_LIST_JSON}}

Core rules:
1. Extract only relations explicitly stated in the text. Do not use world knowledge.
2. Do NOT use any numeric user id or hidden identifiers in relation endpoints.
   When the text uses pronouns/possessives, resolve them only when the owner/subject is explicitly mentioned and unambiguous; otherwise omit that relation.
3. Use only the allowed canonical predicates listed below.
4. Prefer stable, timeless relations. Do NOT create event-like predicates.
   - Good: live_in, work_at, parent_of, spouse_of, has_role
   - Bad: met, called, asked, visited_yesterday, became_professor
5. Questions:
   - Do NOT extract the proposition being asked about.
   - You MAY extract background facts explicitly presupposed by the wording.
     Example: "Does my daughter Alice study at MIT?" -> extract parent_of(USER_ID, Alice), but do NOT extract study_at(Alice, MIT).
6. Do NOT output relations for:
   - future plans or intentions
   - hypotheticals or counterfactuals
   - wishes about future states
   - vague or generic associations
7. Negation / uncertainty:
   - Use factuality="negated" for explicit negation of a stable relation.
   - Use factuality="uncertain" only when the text explicitly states uncertainty about a present or past stable relation.
   - Do NOT use uncertain for plans or hypotheticals; omit those instead.
8. Keep direction consistent with the predicate definitions below.
9. Deduplicate relations. Output each relation only once.
10. Include a short evidence span copied from the text.

Allowed canonical predicates and direction:
- live_in: Person/Organization -> Location
- located_in: non-person entity -> Location
- from: Person -> Location
- born_in: Person -> Location
- work_at: Person -> Organization
- study_at: Person -> Organization
- member_of: Person/Organization -> Organization
- has_role: Person -> Role
- parent_of: parent -> child
- spouse_of: Person -> Person
- sibling_of: Person -> Person
- friend_of: Person -> Person
- reports_to: Person -> Person
- owns: owner -> owned entity
- uses: user -> thing
- likes: Person -> entity/topic
- dislikes: Person -> entity/topic
- interested_in: Person -> Topic
- created_by: CreativeWork/Product -> Person/Organization
- part_of: part -> whole

Additional constraints:
- Use only source/target values that exactly match a `canonical_name` from the provided entity list.
- If no allowed predicate fits cleanly, output nothing for that relation.
- Do not answer the user.


You MUST call the tool function `establish_relationships`.
Provide arguments as a JSON object with key `entities` (array of relation objects).
If no valid relations exist, pass an empty list: {"entities": []}.

Relation item schema (inside the tool arguments):
{
  "source": "canonical_name",
  "target": "canonical_name",
  "predicate_canonical": "live_in|located_in|from|born_in|work_at|study_at|member_of|has_role|parent_of|spouse_of|sibling_of|friend_of|reports_to|owns|uses|likes|dislikes|interested_in|created_by|part_of",
  "predicate_surface": "string",
  "factuality": "asserted|negated|uncertain",
  "evidence": "string"
}
"""




def get_delete_messages(existing_memories_string, data, user_id):
    return DELETE_RELATIONS_SYSTEM_PROMPT.replace(
        "USER_ID", user_id
    ), f"Here are the existing memories: {existing_memories_string} \n\n New Information: {data}"
