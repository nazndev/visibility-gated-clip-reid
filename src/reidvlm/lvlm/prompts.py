"""Versioned pair-comparison prompts. This file is part of the method, not a private note.

``pair-judgment-v1`` is the original prompt. It stays here as a reference.
The executed subset used ``pair-judgment-v2``. The street-photo case is not a
reason to replace either text.
"""

PROMPT_VERSION_V1 = "pair-judgment-v1"
PROMPT_VERSION = "pair-judgment-v2"

PAIR_PROMPT_V1 = """Compare the two photographs. The first photograph is Image A and the second photograph is Image B.
Describe only what is visible. If a cue is occluded, unclear, or not observable, use relation "unknown".
Not visible does not mean that the attribute is absent.
Background, where the person stands in the frame, and camera angle do not establish identity.
Similar clothing alone does not prove identity.
The overall verdict is an uncertain retrieval signal.

Score these five cues: upper_clothing, lower_clothing, footwear, carried_object, distinctive_mark.
For each cue provide visible_in_a, visible_in_b, relation, observation_a, and observation_b.
relation is one of match, mismatch, unknown.
overall_verdict is one of same, different, uncertain.
Keep each observation under 20 words.
Return only one JSON object with those five cue objects and overall_verdict. Do not add prose.
"""

PAIR_PROMPT = """Compare the two photographs. The first photograph is Image A and the second photograph is Image B.
Describe only what is visible. If a cue is occluded, unclear, or not observable, use relation "unknown".
Not visible does not mean that the attribute is absent.
Background, where the person stands in the frame, and camera angle do not establish identity.
Similar clothing alone does not prove identity.
The overall verdict is an uncertain retrieval signal.

Score these five cues: upper_clothing, lower_clothing, footwear, carried_object, distinctive_mark.
For each cue provide visible_in_a, visible_in_b, relation, observation_a, and observation_b.
relation is one of match, mismatch, unknown.
visible_in_a and visible_in_b must be true or false. Use false when the cue is occluded, unclear, or not observable. Do not write unknown in those two fields.
overall_verdict is one of same, different, uncertain.
Keep each observation under 20 words.
Return only one JSON object with those five cue objects and overall_verdict. Do not add prose.
"""

RETRY_NOTE = "The previous reply was not valid JSON for the required schema. Return only the JSON object."
