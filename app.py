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
    st.session_state.messages = [] 
    free_gpu_memory()

# --- 🎯 SEARCH CONFIG ---
def get_search_config(query):
    q_lower = query.lower()
    if "code" in q_lower or "python" in q_lower:
        return {
            "pos": ["software code with indentation", "IDLE python editor window", "programming syntax"],
            "neg": ["presentation slide with bullet points", "powerpoint text slide", "big title text"],
            "penalty": 0.4 
        }
    if "flow" in q_lower or "chart" in q_lower:
        return {
            "pos": ["diagram with boxes and arrows", "flowchart"],
            "neg": ["face of a person", "text paragraph"],
            "penalty": 0.2
        }
    return {"pos": [query], "neg": [], "penalty": 0.0}

# --- 1. INGESTION ---
def process_video_ingestion(video_path):
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text(f"👀 1/2: Building Matrix on {DEVICE.upper()}...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True).to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_data = {"timestamps": [], "embeddings": []}
    frame_count = 0
    sample_rate = int(fps * 2) 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % sample_rate == 0:
            timestamp = frame_count / fps
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            if np.mean(frame) < 30: 
                frame_count += 1; continue
            
            h, w = frame.shape[:2]
            if w > 720: frame = cv2.resize(frame, (720, int(h * (720/w))))
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                embed = model.get_image_features(**inputs)[0]
                embed = embed / embed.norm(p=2, dim=-1, keepdim=True) 
                
            frame_data["timestamps"].append(timestamp)
            frame_data["embeddings"].append(embed.cpu())
        frame_count += 1
    cap.release()
    
    if len(frame_data["embeddings"]) > 0:
        tensor_stack = torch.stack(frame_data["embeddings"])
        torch.save({"timestamps": frame_data["timestamps"], "matrix": tensor_stack}, VECTORS_PATH)
    
    del model, processor; free_gpu_memory()
    progress_bar.empty(); status_text.text(f"✅ Visuals Ready.")
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
    except: return False

@st.cache_resource
def load_resources():
    if os.path.exists(VECTORS_PATH):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True).to(DEVICE)
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
        data = torch.load(VECTORS_PATH)
        return model, proc, data["matrix"].to(DEVICE), data["timestamps"]
    return None, None, None, None

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
        st.rerun()

# --- MAIN APP ---
if "video_processed" not in st.session_state: st.session_state.video_processed = False
if "messages" not in st.session_state: st.session_state.messages = []
if "video_time" not in st.session_state: st.session_state.video_time = 0

# Header
st.title("👁️ Vision-Language Video Analyst")

if not os.path.exists(UPLOAD_PATH) or not os.path.exists(VECTORS_PATH):
    st.info("👈 Upload a video in the sidebar to get started!")
else:
    search_model, search_proc, matrix, timestamps = load_resources()
    full_transcript = ""
    if os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f: full_transcript = f.read()

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
            
            q_lower = query.lower()
            visual_triggers = ["find", "show", "where", "look", "locate", "diagram", "slide", "code", "chart"]
            explain_triggers = ["explain", "what", "why", "define", "summary", "summarize", "how"]
            
            if any(k in q_lower for k in visual_triggers): mode = "VISUAL"
            elif any(k in q_lower for k in explain_triggers): mode = "EXPLAIN"
            else: mode = "HYBRID" 

            with st.spinner(f"Thinking ({mode} Mode)..."):
                unique_matches = []
                
                # --- VISUAL SEARCH ---
                if mode == "VISUAL" or mode == "HYBRID":
                    config = get_search_config(query)
                    
                    inputs_pos = search_proc(text=config["pos"], return_tensors="pt", padding=True).to(DEVICE)
                    with torch.no_grad():
                        embed_pos = search_model.get_text_features(**inputs_pos)
                        embed_pos /= embed_pos.norm(p=2, dim=-1, keepdim=True)
                    
                    embed_neg = None
                    if config["neg"]:
                        inputs_neg = search_proc(text=config["neg"], return_tensors="pt", padding=True).to(DEVICE)
                        with torch.no_grad():
                            embed_neg = search_model.get_text_features(**inputs_neg)
                            embed_neg /= embed_neg.norm(p=2, dim=-1, keepdim=True)

                    pos_scores = torch.matmul(matrix, embed_pos.T)
                    best_pos, _ = torch.max(pos_scores, dim=1) 
                    final_scores = best_pos
                    
                    if embed_neg is not None:
                        neg_scores = torch.matmul(matrix, embed_neg.T)
                        best_neg, _ = torch.max(neg_scores, dim=1)
                        final_scores = best_pos - (config["penalty"] * best_neg)
                    
                    values, indices = torch.topk(final_scores, top_k_slider) 
                    candidates = []
                    for score, idx in zip(values, indices):
                        candidates.append((timestamps[idx.item()], score.item()))
                    
                    candidates.sort(key=lambda x: x[0])
                    if candidates:
                        unique_matches.append(candidates[0])
                        for m in candidates[1:]:
                            if m[0] - unique_matches[-1][0] > 10: unique_matches.append(m)
                    
                    visual_context = f"Matches at: {', '.join([str(int(m[0])) for m in unique_matches])}s"
                else:
                    visual_context = "Concept Search."
                
                st.session_state.last_matches = unique_matches
                
                # --- EXPLAINER ---
                try:
                    if mode == "EXPLAIN":
                        prompt = f"""
                        You are a Professor. Use the transcript below to explain the concept.
                        QUESTION: {query}
                        TRANSCRIPT: {full_transcript[:15000]}
                        INSTRUCTION: Be clear and concise.
                        """
                    else:
                        prompt = f"""
                        You are a Search Assistant.
                        QUERY: {query}
                        VISUALS: {visual_context}
                        INSTRUCTION: Briefly confirm the scenes found.
                        """
                    
                    res = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
                    st.session_state.messages.append({"role": "assistant", "content": res['message']['content']})
                except Exception as e: st.error(f"LLM Error: {e}")
            
        # 3. Found Scenes Grid (Below Input)
        if "last_matches" in st.session_state and st.session_state.last_matches:
            st.caption(f"Found {len(st.session_state.last_matches)} scenes (Most Recent):")
            cols = st.columns(4)
            for idx, (ts, score) in enumerate(st.session_state.last_matches):
                if cols[idx % 4].button(f"▶ {int(ts//60)}:{int(ts%60):02d}", key=f"btn_{idx}"):
                    st.session_state.video_time = int(ts)
                    st.rerun()
            st.divider()

        # 4. Chat History (Stack: Newest First)
        chat_container = st.container()
        with chat_container:
            for m in reversed(st.session_state.messages):
                with st.chat_message(m["role"]):
                    st.write(m["content"])