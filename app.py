import streamlit as st
import ollama
from transformers import CLIPProcessor, CLIPModel
import os
import cv2
from PIL import Image
import numpy as np
from faster_whisper import WhisperModel 
import torch
import gc
from logic import (route_query, get_search_config, penalty_score, clip_embeddings, dedupe_scenes, build_chat_messages,
                   parse_transcript, chunk_segments, retrieve_chunks, format_chunks_for_prompt, format_timestamp)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DATA_DIR = os.path.join(BASE_DIR, "video_data")
UPLOAD_PATH = os.path.join(VIDEO_DATA_DIR, "current_video.mp4")
TRANSCRIPT_PATH = os.path.join(VIDEO_DATA_DIR, "transcript.txt")
VECTORS_PATH = os.path.join(VIDEO_DATA_DIR, "video_vectors.pt")

os.makedirs(VIDEO_DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Vision-Language Video Analyst", layout="wide", page_icon="👁️")

# --- 🎨 CUSTOM CSS: UI POLISH ---
st.markdown("""
<style>
    /* 1. PILL FORM STYLING */
    div[data-testid="stForm"] {
        background-color: #262730;
        border: 1px solid #4A4A4A;
        border-radius: 30px;
        padding: 5px 10px;
    }

    /* 2. TRANSPARENT INPUT & BUTTON */
    div[data-testid="stTextInput"] input {
        background-color: transparent !important;
        border: none !important;
        color: white !important;
        padding-left: 10px;
    }
    div[data-testid="stTextInput"] input:focus {
        box-shadow: none !important;
    }
    div[data-testid="stFormSubmitButton"] button {
        background-color: transparent !important;
        border: none !important;
        color: white !important;
        padding: 0px !important;
        font-size: 20px !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        color: #00BFFF !important;
    }

    /* 3. FIX CLEAR CHAT BUTTON TEXT VISIBILITY */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        color: #FFFFFF !important; /* Force White Text */
        border-color: #4A4A4A !important;
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
    }

    /* 4. REDUCE GAPS (TIGHTER LAYOUT) */
    .block-container {
        padding-top: 2rem !important; /* Less top padding */
        padding-bottom: 1rem !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem !important; /* Shrink vertical gap between elements */
    }
    div[data-testid="column"] {
        padding: 0px !important;
    }
    /* Hide label containers */
    div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- HARDWARE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "int8" 

# --- HELPERS ---
def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def reset_system():
    if os.path.exists(VECTORS_PATH): os.remove(VECTORS_PATH)
    if os.path.exists(TRANSCRIPT_PATH): os.remove(TRANSCRIPT_PATH)
    st.session_state.video_processed = False
    st.session_state.pop('last_matches', None)
    st.session_state.pop('last_sources', None)
    st.session_state.messages = []
    st.session_state.video_time = 0
    st.session_state.ingest_errors = []
    load_index.clear()
    free_gpu_memory()

def report_ingest_error(msg):
    # Shown now and kept in session state so it survives the st.rerun() after analysis.
    st.error(msg)
    st.session_state.setdefault("ingest_errors", []).append(msg)

def ollama_error_message(e):
    text = str(e)
    if getattr(e, "status_code", None) == 404 or "not found" in text.lower():
        return "The llama3 model isn't available in Ollama. Run `ollama pull llama3`, then try again."
    if isinstance(e, ConnectionError) or "connect" in type(e).__name__.lower() or "connect" in text.lower():
        return "Can't reach the Ollama server. Start it with `ollama serve` (and run `ollama pull llama3` once), then try again."
    return f"LLM Error: {e}"

# --- 1. INGESTION ---
def process_video_ingestion(video_path):
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text(f"👀 1/2: Building Matrix on {DEVICE.upper()}...")
    
    model, processor = load_clip()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release(); progress_bar.empty(); status_text.empty()
        report_ingest_error("Could not read this video (no frame rate found). Is it a valid MP4?")
        return False

    frame_data = {"timestamps": [], "embeddings": []}
    frame_count = 0
    sample_rate = max(int(fps * 2), 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % sample_rate == 0:
            timestamp = frame_count / fps
            progress_bar.progress(min(frame_count / max(total_frames, 1), 1.0))
            if np.mean(frame) < 30: 
                frame_count += 1; continue
            
            h, w = frame.shape[:2]
            if w > 720: frame = cv2.resize(frame, (720, int(h * (720/w))))
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                embed = clip_embeddings(model.get_image_features(**inputs))[0]
                embed = embed / embed.norm(p=2, dim=-1, keepdim=True) 
                
            frame_data["timestamps"].append(timestamp)
            frame_data["embeddings"].append(embed.cpu())
        frame_count += 1
    cap.release()
    
    free_gpu_memory()
    progress_bar.empty()
    if not frame_data["embeddings"]:
        status_text.empty()
        report_ingest_error("No frames could be indexed: the video is empty or every sampled frame is too dark. Visual search is unavailable for this video.")
        return False

    tensor_stack = torch.stack(frame_data["embeddings"])
    torch.save({"timestamps": frame_data["timestamps"], "matrix": tensor_stack}, VECTORS_PATH)
    status_text.text(f"✅ Visuals Ready.")
    return True

def process_audio_transcription(video_path, model_size):
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text(f"⚡ 2/2: Listening...")
    try:
        model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
        segments, info = model.transcribe(video_path, beam_size=5)
        with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            for s in segments:
                progress_bar.progress(min(s.end / info.duration, 1.0))
                if s.no_speech_prob < 0.4: f.write(f"[{int(s.start)}] {s.text}\n")
        del model; free_gpu_memory()
        status_text.text("✅ Audio Done!"); progress_bar.empty()
        return True
    except Exception as e:
        progress_bar.empty(); status_text.empty(); free_gpu_memory()
        # A partial transcript would silently skew retrieval, so drop it entirely.
        if os.path.exists(TRANSCRIPT_PATH): os.remove(TRANSCRIPT_PATH)
        report_ingest_error(f"Transcription failed ({type(e).__name__}: {e}). Visual search still works, but explanations have no transcript to draw on.")
        return False

@st.cache_resource
def load_clip():
    # The model never changes between videos, so it is loaded once per process.
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True).to(DEVICE)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
    return model, proc

@st.cache_resource(max_entries=1)
def load_index(vectors_mtime):
    # Keyed on the file's mtime so a newly analyzed video is never served from a stale cache.
    data = torch.load(VECTORS_PATH, weights_only=True)
    return data["matrix"].to(DEVICE), data["timestamps"]

def encode_text(model, proc, texts):
    inputs = proc(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embed = clip_embeddings(model.get_text_features(**inputs))
    return embed / embed.norm(p=2, dim=-1, keepdim=True)

@st.cache_data(max_entries=1)
def load_transcript_chunks(transcript_mtime):
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        return chunk_segments(parse_transcript(f.read()))

def render_jump_buttons(caption, times, key_prefix):
    st.caption(caption)
    cols = st.columns(4)
    for idx, ts in enumerate(times):
        if cols[idx % 4].button(f"▶ {format_timestamp(ts)}", key=f"btn_{key_prefix}_{idx}_{ts:.2f}"):
            st.session_state.video_time = int(ts)
            st.rerun()

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 🧠 Audio Intelligence")
    model_size = st.selectbox("Whisper Model", ["small", "medium"], index=0)
    
    st.markdown("### 📂 Upload Video")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["mp4"])
    
    if uploaded_file and st.button("🚀 Analyze Video", type="primary"):
        with open(UPLOAD_PATH, "wb") as f: f.write(uploaded_file.getbuffer())
        reset_system()
        process_video_ingestion(UPLOAD_PATH)
        process_audio_transcription(UPLOAD_PATH, model_size)
        st.session_state.video_processed = True
        st.rerun()

    st.markdown("### 🔍 Search Sensitivity")
    top_k_slider = st.slider("Max Scenes", min_value=1, max_value=20, value=8)
    
    st.divider()
    
    # 🧹 Clear Chat Button (Text visibility fixed via CSS)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.pop('last_matches', None)
        st.session_state.pop('last_sources', None)
        st.rerun()

# --- MAIN APP ---
if "video_processed" not in st.session_state: st.session_state.video_processed = False
if "messages" not in st.session_state: st.session_state.messages = []
if "video_time" not in st.session_state: st.session_state.video_time = 0

# Header
st.title("👁️ Vision-Language Video Analyst")

for err in st.session_state.get("ingest_errors", []):
    st.error(err)

if not os.path.exists(UPLOAD_PATH) or not os.path.exists(VECTORS_PATH):
    if not st.session_state.get("ingest_errors"):
        st.info("👈 Upload a video in the sidebar to get started!")
else:
    search_model, search_proc = load_clip()
    matrix, timestamps = load_index(os.path.getmtime(VECTORS_PATH))
    transcript_chunks = []
    if os.path.exists(TRANSCRIPT_PATH):
        transcript_chunks = load_transcript_chunks(os.path.getmtime(TRANSCRIPT_PATH))

    # --- LAYOUT: 2 Columns ---
    col1, col2 = st.columns([1.6, 1])
    
    # 🎥 LEFT COLUMN: Video Player
    with col1:
        st.markdown("### 📺 Video Player")
        with open(UPLOAD_PATH, 'rb') as f: 
            st.video(f.read(), start_time=st.session_state.video_time)

    # 💬 RIGHT COLUMN: Chat & Search
    with col2:
        st.markdown("### 🗨️ Chat & Search")
        
        # 1. Visual Context Box (Blue)
        st.info("📌 **Visual Context:** System Ready. Video processed and indexed. Ask about visuals or concepts.")
        
        # 2. Text ABOVE the Search Bar (Restored as requested)
        st.caption("Ask: 'Find python code' or 'Summarize the video'")

        # 3. Input Box (Pill Style)
        with st.form(key="search_form", clear_on_submit=False): 
            c_in, c_btn = st.columns([0.85, 0.15])
            with c_in:
                # Text INSIDE the pill (Placeholder)
                query = st.text_input(
                    "Search", 
                    placeholder="Ask: 'Summarize the video' or 'Find code'", 
                    label_visibility="collapsed"
                )
            with c_btn:
                # The '➤' Arrow Button
                submit_btn = st.form_submit_button("➤")

        # Logic Trigger
        if submit_btn and query:
            st.session_state.messages.append({"role": "user", "content": query})
            
            mode = route_query(query)

            with st.spinner(f"Thinking ({mode} Mode)..."):
                unique_matches = []
                
                # --- VISUAL SEARCH ---
                if mode == "VISUAL" or mode == "HYBRID":
                    config = get_search_config(query)

                    embed_pos = encode_text(search_model, search_proc, config["pos"])
                    best_pos, _ = torch.max(torch.matmul(matrix, embed_pos.T), dim=1)

                    best_neg = None
                    if config["neg"]:
                        embed_neg = encode_text(search_model, search_proc, config["neg"])
                        best_neg, _ = torch.max(torch.matmul(matrix, embed_neg.T), dim=1)

                    query_sim = None
                    if config["query"]:
                        embed_query = encode_text(search_model, search_proc, [config["query"]])
                        query_sim = torch.matmul(matrix, embed_query.T)[:, 0]

                    final_scores = penalty_score(best_pos, best_neg, query_sim, config)

                    k = min(top_k_slider, matrix.shape[0])
                    values, indices = torch.topk(final_scores, k)
                    candidates = []
                    for score, idx in zip(values, indices):
                        candidates.append((timestamps[idx.item()], score.item()))

                    unique_matches = dedupe_scenes(candidates, min_gap=10)

                    visual_context = "Matches at: " + (", ".join(f"[{format_timestamp(m[0])}]" for m in unique_matches) or "none")
                else:
                    visual_context = "Concept Search."
                
                st.session_state.last_matches = unique_matches

                # --- TRANSCRIPT RETRIEVAL ---
                sources = retrieve_chunks(query, transcript_chunks, top_n=5) if mode in ("EXPLAIN", "HYBRID") else []
                st.session_state.last_sources = sources
                transcript_context = format_chunks_for_prompt(sources) or "(No transcript available.)"
                cite = "Cite the timestamps you rely on in the form [m:ss], e.g. [2:15]."

                # --- EXPLAINER ---
                try:
                    if mode == "EXPLAIN":
                        prompt = f"""
                        You are a Professor. Use the transcript excerpts below to explain the concept.
                        QUESTION: {query}
                        TRANSCRIPT EXCERPTS:
                        {transcript_context}
                        INSTRUCTION: Be clear and concise. {cite}
                        """
                    elif mode == "HYBRID":
                        prompt = f"""
                        You are a Video Assistant. Answer using both the visual matches and the transcript.
                        QUESTION: {query}
                        VISUALS: {visual_context}
                        TRANSCRIPT EXCERPTS:
                        {transcript_context}
                        INSTRUCTION: Be clear and concise, and mention the relevant scenes. {cite}
                        """
                    else:
                        prompt = f"""
                        You are a Search Assistant.
                        QUERY: {query}
                        VISUALS: {visual_context}
                        INSTRUCTION: Briefly confirm the scenes found, citing their timestamps as given (e.g. [2:15]).
                        """
                    
                    # Prior turns exclude the query just appended above; it is sent as `prompt` with its context.
                    history = st.session_state.messages[:-1]
                    res = ollama.chat(model='llama3', messages=build_chat_messages(history, prompt, max_history=6))
                    st.session_state.messages.append({"role": "assistant", "content": res['message']['content']})
                except Exception as e: st.error(ollama_error_message(e))
            
        # 3. Found Scenes Grid (Below Input)
        if st.session_state.get("last_matches"):
            render_jump_buttons(f"Found {len(st.session_state.last_matches)} scenes (in chronological order):",
                                [ts for ts, _ in st.session_state.last_matches], "scene")
        if st.session_state.get("last_sources"):
            render_jump_buttons(f"Transcript sources ({len(st.session_state.last_sources)}):",
                                [c["start"] for c in st.session_state.last_sources], "source")
        if st.session_state.get("last_matches") or st.session_state.get("last_sources"):
            st.divider()

        # 4. Chat History (Stack: Newest First)
        chat_container = st.container()
        with chat_container:
            for m in reversed(st.session_state.messages):
                with st.chat_message(m["role"]):
                    st.write(m["content"])