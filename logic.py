"""Pure query/transcript logic for app.py.

Kept free of Streamlit, CLIP and Ollama imports so it can be unit tested
(see tests/test_logic.py) without loading models or starting servers.
"""
import re


def _word_pattern(words):
    """Regex matching any of `words` (single words or phrases) as whole words."""
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b")


def _normalize(text):
    # Curly apostrophes (common from phone keyboards) would break "what's".
    return text.lower().replace("’", "'")


# --- 🧭 ROUTER ---
EXPLAIN_PATTERN = re.compile(
    r"\b(?:explain\w*|why|how|define\w*|definition|summari[sz]e\w*|summary"
    r"|what\s+(?:is|are)|what's)\b"
)
VISUAL_PATTERN = _word_pattern([
    "find", "show", "where", "locate", "look",
    "diagram", "slide", "code", "chart",
])


def route_query(query):
    """Return "EXPLAIN", "VISUAL" or "HYBRID" for a user query.

    Explain verbs take precedence, so "explain how the code works" is EXPLAIN
    even though it mentions "code". Visual words win only when no explain
    verb is present; anything else falls back to HYBRID.
    """
    q = _normalize(query)
    if EXPLAIN_PATTERN.search(q):
        return "EXPLAIN"
    if VISUAL_PATTERN.search(q):
        return "VISUAL"
    return "HYBRID"
