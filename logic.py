"""Pure query/transcript logic for app.py.

Kept free of Streamlit, CLIP and Ollama imports so it can be unit tested
(see tests/test_logic.py) without loading models or starting servers.
"""
import math
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



# --- 🎯 SEARCH CONFIG ---
# topic -> keywords (whole words/phrases), CLIP positive prompts, negative
# prompts and penalty, for Score = Max(Positive) - penalty * Max(Negative).
# The first topic whose keywords match wins, so more specific topics (e.g.
# "bar chart" under graph) are listed before broader ones ("chart").
SEARCH_CONFIGS = {
    "code": {
        "keywords": ["code", "coding", "python", "java", "javascript", "programming",
                     "program", "script", "editor", "ide", "terminal", "syntax"],
        "pos": ["software code with indentation", "IDLE python editor window", "programming syntax"],
        "neg": ["presentation slide with bullet points", "powerpoint text slide", "big title text"],
        "penalty": 0.4,
    },
    "graph": {
        "keywords": ["graph", "graphs", "plot", "plots", "bar chart", "line chart", "pie chart",
                     "histogram", "axis", "axes", "curve"],
        "pos": ["a plotted graph with x and y axes", "a bar chart", "a line plot of data"],
        "neg": ["paragraph of text", "face of a person"],
        "penalty": 0.2,
    },
    "flowchart": {
        "keywords": ["flow", "flowchart", "flowcharts", "flow chart", "chart", "diagram",
                     "diagrams", "arrows", "uml"],
        "pos": ["diagram with boxes and arrows", "flowchart"],
        "neg": ["face of a person", "text paragraph"],
        "penalty": 0.2,
    },
    "whiteboard": {
        "keywords": ["whiteboard", "blackboard", "chalkboard", "board", "handwriting",
                     "handwritten", "chalk", "marker"],
        "pos": ["handwritten notes on a whiteboard", "chalk writing on a blackboard",
                "a person writing on a board"],
        "neg": ["typed presentation slide", "computer screen"],
        "penalty": 0.3,
    },
    "table": {
        "keywords": ["table", "tables", "spreadsheet", "rows", "columns", "grid"],
        "pos": ["a table with rows and columns", "a spreadsheet grid of numbers"],
        "neg": ["paragraph of text", "slide with bullet points"],
        "penalty": 0.2,
    },
    "equation": {
        "keywords": ["equation", "equations", "formula", "formulas", "math", "maths",
                     "mathematics", "derivation", "proof", "integral", "theorem"],
        "pos": ["a mathematical equation", "math formula with symbols", "handwritten math derivation"],
        "neg": ["software code", "photo of a person"],
        "penalty": 0.25,
    },
    "person": {
        "keywords": ["person", "people", "speaker", "lecturer", "professor", "instructor",
                     "teacher", "presenter", "face"],
        "pos": ["a person speaking to the camera", "a lecturer standing in front of a class",
                "face of a presenter"],
        "neg": ["presentation slide with text", "computer screen with code"],
        "penalty": 0.3,
    },
}
_TOPIC_PATTERNS = {topic: _word_pattern(cfg["keywords"]) for topic, cfg in SEARCH_CONFIGS.items()}


def match_topic(query):
    """Return the first SEARCH_CONFIGS topic whose keywords appear as whole words, else None."""
    q = _normalize(query)
    for topic, pattern in _TOPIC_PATTERNS.items():
        if pattern.search(q):
            return topic
    return None


def get_search_config(query):
    """Return {"pos", "neg", "penalty"} for the visual search of `query`.

    Unmatched queries always use the raw query as the positive prompt (no penalty).
    """
    topic = match_topic(query)
    if topic is None:
        return {"pos": [query], "neg": [], "penalty": 0.0}
    cfg = SEARCH_CONFIGS[topic]
    return {"pos": list(cfg["pos"]), "neg": list(cfg["neg"]), "penalty": cfg["penalty"]}


def dedupe_scenes(candidates, min_gap=10):
    """Sort (timestamp, score) matches by time and drop any within min_gap seconds of the last kept one."""
    unique = []
    for m in sorted(candidates, key=lambda x: x[0]):
        if not unique or m[0] - unique[-1][0] > min_gap:
            unique.append(m)
    return unique

# --- 📜 TRANSCRIPT RETRIEVAL ---
# Retrieval uses BM25 keyword scoring in pure Python rather than CLIP's text
# encoder: CLIP truncates at 77 tokens (~1/3 of an 800-char chunk) and was
# trained on short image captions, so it embeds lecture prose poorly. BM25
# sees the whole chunk, needs no extra dependency and is easy to test.
SEGMENT_PATTERN = re.compile(r"^\[(\d+(?:\.\d+)?)\]\s*(.*)$")
STOPWORDS = frozenset("""
a an and are as at be but by can did do does for from had has have he her his
how i if in into is it its me my of on or our she so that the their them then
there these they this to was we were what when where which who why will with
you your about explain explains define summarize summarise summary tell please
video lecture talk speaker say says said mean means
""".split())


def format_timestamp(seconds):
    """Format seconds as m:ss (e.g. 135 -> "2:15")."""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def parse_transcript(text):
    """Parse "[sec] text" lines into [(start_seconds, text)], skipping malformed lines."""
    segments = []
    for line in text.splitlines():
        m = SEGMENT_PATTERN.match(line.strip())
        if m and m.group(2).strip():
            segments.append((float(m.group(1)), m.group(2).strip()))
    return segments


def chunk_segments(segments, max_seconds=60, max_chars=800):
    """Group consecutive segments into chunks of ~max_seconds or ~max_chars.

    Returns [{"start": seconds, "text": str}], where start is the start time
    of the chunk's first segment.
    """
    chunks = []
    for start, text in segments:
        if chunks:
            cur = chunks[-1]
            fits = (start - cur["start"] < max_seconds
                    and len(cur["text"]) + 1 + len(text) <= max_chars)
            if fits:
                cur["text"] += " " + text
                continue
        chunks.append({"start": start, "text": text})
    return chunks


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", _normalize(text)) if t not in STOPWORDS]


def retrieve_chunks(query, chunks, top_n=5, k1=1.5, b=0.75):
    """Return the top_n chunks for `query` by BM25, in chronological order.

    If no query term appears in the transcript (e.g. "summarize the video"),
    fall back to chunks spread evenly across the video so the LLM still gets
    an overview.
    """
    if not chunks:
        return []
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1

    terms = set(tokenize(query))
    scores = []
    for i, d in enumerate(docs):
        score = 0.0
        for term in terms:
            tf = d.count(term)
            if tf:
                idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1)
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
        scores.append((score, i))

    if any(s > 0 for s, _ in scores):
        ranked = sorted(scores, key=lambda x: (-x[0], x[1]))
        picked = [i for s, i in ranked[:top_n] if s > 0]
    else:
        k = min(top_n, n)
        picked = sorted({round(j * (n - 1) / max(k - 1, 1)) for j in range(k)})
    return [chunks[i] for i in sorted(picked)]


def format_chunks_for_prompt(chunks):
    return "\n".join(f"[{format_timestamp(c['start'])}] {c['text']}" for c in chunks)


# --- 💬 CONVERSATION MEMORY ---
def build_chat_messages(history, prompt, max_history=6):
    """Return Ollama chat messages: the last `max_history` prior turns, then `prompt` as the user turn."""
    prior = history[-max_history:] if max_history > 0 else []
    return [{"role": m["role"], "content": m["content"]} for m in prior] + [{"role": "user", "content": prompt}]
