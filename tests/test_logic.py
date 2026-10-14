import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic import (  # noqa: E402
    build_chat_messages,
    chunk_segments,
    clip_embeddings,
    dedupe_scenes,
    format_chunks_for_prompt,
    format_timestamp,
    get_search_config,
    QUERY_WEIGHT,
    match_topic,
    parse_transcript,
    penalty_score,
    retrieve_chunks,
    route_query,
)


# --- route_query ---

@pytest.mark.parametrize("query, mode", [
    # The original bug: "code" must not pull an explain query into VISUAL.
    ("explain how the code works", "EXPLAIN"),
    ("Explain the algorithm", "EXPLAIN"),
    ("why is the slide red", "EXPLAIN"),
    ("Show me how recursion works", "EXPLAIN"),
    ("define a closure", "EXPLAIN"),
    ("Summarize the video", "EXPLAIN"),
    ("give me a summary", "EXPLAIN"),
    ("what is a linked list", "EXPLAIN"),
    ("What are pointers?", "EXPLAIN"),
    ("What’s a monad", "EXPLAIN"),
    ("Find python code", "VISUAL"),
    ("show the flow chart", "VISUAL"),
    ("Where is the diagram?", "VISUAL"),
    ("locate the whiteboard", "VISUAL"),
    ("look for the title slide", "VISUAL"),
    ("recursion", "HYBRID"),
    ("binary trees", "HYBRID"),
    ("", "HYBRID"),
])
def test_route_query(query, mode):
    assert route_query(query) == mode


@pytest.mark.parametrize("query", [
    "barcode scanner",   # contains "code"
    "showcase",          # contains "show"
    "somehow done",      # contains "how"
    "whatever it is",    # contains "what" ... "is" but not "what is"
    "nowhere",           # contains "where"
])
def test_route_query_matches_whole_words_only(query):
    assert route_query(query) == "HYBRID"


# --- get_search_config ---

@pytest.mark.parametrize("query, topic", [
    ("Find python code", "code"),
    ("show the IDE", "code"),
    ("show me the flow chart", "flowchart"),
    ("find the chart", "flowchart"),
    ("show the bar chart", "graph"),
    ("find the plot of loss", "graph"),
    ("handwriting on the whiteboard", "whiteboard"),
    ("show the results table", "table"),
    ("where is the equation", "equation"),
    ("find the math", "equation"),
    ("show the speaker", "person"),
    ("FIND THE PROFESSOR", "person"),
])
def test_match_topic(query, topic):
    assert match_topic(query) == topic


@pytest.mark.parametrize("query", ["barcode", "decoder ring", "flowing water", "tablet", "find a cat"])
def test_match_topic_requires_whole_words(query):
    assert match_topic(query) is None


def test_get_search_config_for_topic():
    cfg = get_search_config("find python code")
    assert "software code with indentation" in cfg["pos"]
    assert cfg["neg"]
    assert cfg["penalty"] == 0.4


def test_get_search_config_topic_adds_raw_query_at_lower_weight():
    cfg = get_search_config("code for sorting")
    assert cfg["query"] == "code for sorting"
    assert cfg["query"] not in cfg["pos"]  # kept out of the max-pooled topic prompts
    assert 0 < cfg["query_weight"] < 1
    assert cfg["query_weight"] == QUERY_WEIGHT == 0.5


def test_get_search_config_unmatched_uses_raw_query():
    cfg = get_search_config("a cat on a sofa")
    assert cfg == {"pos": ["a cat on a sofa"], "neg": [], "penalty": 0.0, "query": None, "query_weight": 0.0}


def test_penalty_score_formula():
    cfg = {"penalty": 0.4, "query_weight": 0.5}
    assert penalty_score(0.30, 0.20, 0.10, cfg) == pytest.approx(0.30 + 0.5 * 0.10 - 0.4 * 0.20)
    assert penalty_score(0.30, None, None, cfg) == pytest.approx(0.30)
    assert penalty_score(0.30, 0.9, 0.9, {"penalty": 0.0, "query_weight": 0.0}) == pytest.approx(0.30)


def test_penalty_score_query_breaks_ties_but_topic_dominates():
    cfg = get_search_config("code for sorting")
    # Two code-editor frames with equal topic scores: the one matching "sorting" wins.
    sorting_editor = penalty_score(0.30, 0.10, 0.28, cfg)
    other_editor = penalty_score(0.30, 0.10, 0.20, cfg)
    assert sorting_editor > other_editor
    # A slide that only mentions "code for sorting" scores well on the raw query but
    # poorly on the topic prompts and hits the slide penalty, so it doesn't win.
    slide_about_it = penalty_score(0.22, 0.30, 0.32, cfg)
    assert other_editor > slide_about_it


def test_get_search_config_returns_copies():
    get_search_config("find code")["pos"].append("mutated")
    assert "mutated" not in get_search_config("find code")["pos"]


# --- transcript parsing / chunking ---

TRANSCRIPT = """[0]  Welcome to the lecture on sorting.
[5] Today we cover merge sort and quicksort.
not a transcript line
[70] Recursion is when a function calls itself.

[75.5] The base case stops the recursion.
[80]
[140] Quicksort picks a pivot element.
"""


def test_parse_transcript():
    assert parse_transcript(TRANSCRIPT) == [
        (0.0, "Welcome to the lecture on sorting."),
        (5.0, "Today we cover merge sort and quicksort."),
        (70.0, "Recursion is when a function calls itself."),
        (75.5, "The base case stops the recursion."),
        (140.0, "Quicksort picks a pivot element."),
    ]


def test_parse_transcript_handles_crlf_and_empty():
    assert parse_transcript("[3] hi\r\n[4] there\r\n") == [(3.0, "hi"), (4.0, "there")]
    assert parse_transcript("") == []


def test_chunk_segments_splits_on_time():
    chunks = chunk_segments(parse_transcript(TRANSCRIPT), max_seconds=60)
    assert [c["start"] for c in chunks] == [0.0, 70.0, 140.0]
    assert chunks[0]["text"] == "Welcome to the lecture on sorting. Today we cover merge sort and quicksort."
    assert chunks[1]["text"] == "Recursion is when a function calls itself. The base case stops the recursion."


def test_chunk_segments_splits_on_chars():
    segments = [(float(i), "x" * 30) for i in range(10)]
    chunks = chunk_segments(segments, max_seconds=60, max_chars=100)
    assert all(len(c["text"]) <= 100 for c in chunks)
    assert [c["start"] for c in chunks] == [0.0, 3.0, 6.0, 9.0]
    assert sum(c["text"].count("x" * 30) for c in chunks) == 10


def test_chunk_segments_keeps_oversized_segment():
    chunks = chunk_segments([(0.0, "y" * 2000), (1.0, "z")], max_chars=800)
    assert [c["start"] for c in chunks] == [0.0, 1.0]


def test_chunk_segments_empty():
    assert chunk_segments([]) == []


# --- retrieval ---

CHUNKS = [
    {"start": 0.0, "text": "Welcome. Today we cover merge sort and quicksort."},
    {"start": 60.0, "text": "Recursion is when a function calls itself. The base case stops recursion."},
    {"start": 120.0, "text": "Quicksort picks a pivot and partitions the array."},
    {"start": 180.0, "text": "Hash tables give constant time lookups."},
    {"start": 240.0, "text": "Thanks for watching, see you next week."},
]


def test_retrieve_chunks_ranks_relevant_chunk():
    top = retrieve_chunks("explain recursion and the base case", CHUNKS, top_n=1)
    assert [c["start"] for c in top] == [60.0]


def test_retrieve_chunks_only_returns_matches_in_time_order():
    top = retrieve_chunks("how does quicksort work", CHUNKS, top_n=5)
    assert [c["start"] for c in top] == [0.0, 120.0]


def test_retrieve_chunks_falls_back_to_spread_for_no_overlap():
    top = retrieve_chunks("summarize the video", CHUNKS, top_n=3)
    assert [c["start"] for c in top] == [0.0, 120.0, 240.0]


def test_retrieve_chunks_edge_cases():
    assert retrieve_chunks("anything", [], top_n=5) == []
    assert len(retrieve_chunks("summarize", CHUNKS[:2], top_n=5)) == 2


def test_format_timestamp_and_prompt():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(135.9) == "2:15"
    assert format_timestamp(3725) == "62:05"
    assert format_chunks_for_prompt(CHUNKS[:2]).splitlines()[1].startswith("[1:00] Recursion")


# --- CLIP output compatibility ---

def test_clip_embeddings_handles_tensor_and_output_object():
    class FakeTensor:  # transformers 4.x: the projected embedding itself
        pass

    class FakeOutput:  # transformers 5.x: BaseModelOutputWithPooling
        def __init__(self, pooled):
            self.last_hidden_state = object()
            self.pooler_output = pooled

    tensor = FakeTensor()
    assert clip_embeddings(tensor) is tensor
    assert clip_embeddings(FakeOutput(tensor)) is tensor


# --- scene de-duplication ---

def test_dedupe_scenes_sorts_and_drops_close_matches():
    candidates = [(30.0, 0.5), (0.0, 0.9), (8.0, 0.8), (10.0, 0.1), (11.0, 0.2), (21.5, 0.3)]
    assert dedupe_scenes(candidates, min_gap=10) == [(0.0, 0.9), (11.0, 0.2), (21.5, 0.3)]


def test_dedupe_scenes_gap_is_strict():
    assert dedupe_scenes([(0.0, 1.0), (10.0, 1.0)], min_gap=10) == [(0.0, 1.0)]
    assert dedupe_scenes([(0.0, 1.0), (10.5, 1.0)], min_gap=10) == [(0.0, 1.0), (10.5, 1.0)]


def test_dedupe_scenes_empty():
    assert dedupe_scenes([]) == []


# --- conversation memory ---

def test_build_chat_messages_keeps_last_turns():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(10)]
    msgs = build_chat_messages(history, "PROMPT", max_history=6)
    assert [m["content"] for m in msgs] == ["4", "5", "6", "7", "8", "9", "PROMPT"]
    assert msgs[-1]["role"] == "user"
    assert build_chat_messages(history, "P", max_history=0) == [{"role": "user", "content": "P"}]
