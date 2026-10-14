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
