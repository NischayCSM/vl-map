# VL-MaP: Vision-Language Matrix Penalty Search

**An Intelligent Multimodal Video Analysis System**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![CLIP](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F16-green) 

## Overview

**VL-MaP** (Vision-Language Matrix Penalty) is an intelligent video analysis tool designed to perform precise, semantic retrieval on educational and technical video content.

Standard vector search tools often struggle with "visual synonyms" in lecture videos—for example, confusing a PowerPoint slide *about* code with an actual *code editor* interface. **VL-MaP** solves this by introducing a novel **Matrix-Penalty Architecture**. It combines a positive search vector (what you want) with a negative penalty vector (what you don't want) to mathematically filter out false positives in real-time.

The system features a **Smart Router** that distinguishes between visual queries ("Find code") and conceptual queries ("Explain recursion"). Conceptual queries go to a Retrieval-Augmented Generation (RAG) engine: the most relevant transcript passages are retrieved with BM25 and given to **Llama 3** together with their timestamps.

## Key Features

* **Matrix-Penalty Visual Search:**
    * Uses **CLIP (ViT-B/16 Patch16)** for high-resolution visual embedding.
    * Implements a custom linear algebra search engine using direct tensor operations (eliminating the need for a vector database).
    * Applies a dynamic penalty score equation:  
      $$Score = Max(Positive) - (Penalty \times Max(Negative))$$
    * Positive/negative prompts and penalties come from a topic table (code, graph/plot, flowchart/diagram, whiteboard, table, equation/math, person/speaker). Queries that match no topic search for the raw query text with no penalty.
* **Intelligent RAG Explainer:**
    * Uses **Faster-Whisper** for high-accuracy speech-to-text transcription.
    * Splits the transcript into ~60-second / ~800-character chunks, each tagged with its start time.
    * Ranks chunks against the question with a lightweight pure-Python **BM25** keyword scorer (no vector database, no extra dependencies) and sends the top 5 to **Llama 3** (via Ollama), which cites timestamps like `[2:15]`. Questions with no keyword overlap (e.g. "Summarize the video") get passages sampled evenly across the video instead.
    * The retrieved passages also appear as clickable ▶ timestamp buttons.
    * The last 6 chat messages are passed to Llama, so follow-up questions keep their context.
* **Smart Intent Router:**
    * Matches whole words in the query and picks one of three modes:
        * **Explain** if it contains an explain word (*explain, why, how, define, summarize/summary, what is/are*). These take precedence, so "Explain how the code works" is Explain, not Visual.
        * **Visual** if it contains a visual word (*find, show, where, locate, look, diagram, slide, code, chart*) and no explain word. This runs only the matrix-penalty search.
        * **Hybrid** for everything else. It runs the visual search *and* transcript retrieval and gives Llama both.
* **Efficient Architecture:**
    * **Phase 1 (Ingestion):** Pre-computes video matrices and transcripts once per video.
    * **Phase 2 (Inference):** Runs searches in milliseconds using cached tensors loaded in VRAM.
    * **Stack-Based UI:** Matched scenes are listed in chronological order, and chat history is stacked with the newest message on top.

## Architecture

The system operates in two distinct phases:

1.  **Ingestion:** Converts raw video frames and audio into a **Video Matrix** (`.pt` file) and a **Transcript** (`.txt` file).
2.  **Inference:** Performs dot-product similarity search with penalty logic for visuals, BM25 transcript retrieval + Llama 3 for concepts, or both (Hybrid).

## Installation

### Prerequisites
* **Python 3.10+**
* **Ollama** (Must be installed and running locally)
* **FFmpeg** (Required for audio extraction)
* **GPU** (Recommended: CUDA for fast inference)

### Clone the Repository

```bash
git clone https://github.com/NischayCSM/vl-map.git
cd vl-map
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** FFmpeg and Ollama are system installs, not Python packages, so `pip` will not install them. Install FFmpeg from https://ffmpeg.org (or your OS package manager) and Ollama from https://ollama.com. For CUDA builds of PyTorch, follow the selector at https://pytorch.org before running the command above.
### Setup Ollama

Download and run Ollama, then pull the Llama 3 model:
```bash
ollama pull llama3
```
# Usage

### Run the Application

```bash
streamlit run app.py
```
### Analyze a Video

* Open the web interface (default: http://localhost:8501).
* Upload an MP4 video file using the sidebar.
* Click "🚀 Analyze".
  The system will process visuals (CLIP) and audio (Whisper). This takes a few minutes depending on video length and GPU power.

### Search & Chat

* Visual Search: Type queries like "Find python code", "Show me the flow chart".
    * The system will return a list of clickable timestamps leading to the exact frame.
* Conceptual Explanation: Type queries like "Explain the algorithm", "Summarize the intro".
    * The system retrieves the most relevant transcript passages and gives a professor-style explanation that cites timestamps. Click a source's ▶ button to jump there.

## Project Structure

```
vl-map/
├── app.py # Main Streamlit Application (The "Intelligent System")
├── logic.py # Pure logic: router, search configs, transcript chunking/BM25, scene de-dup
├── tests/
│ └── test_logic.py # Unit tests for logic.py (pytest)
├── requirements.txt # Python Dependencies
├── video_data/ # Generated Assets (Gitignored)
│ ├── current_video.mp4
│ ├── video_vectors.pt # The Video Matrix (Tensor File)
│ └── transcript.txt # Speech-to-Text Output
└── README.md # Documentation
```

## Running Tests

The tests cover the pure logic in `logic.py` and don't load CLIP, Whisper, Streamlit or Ollama:

```bash
python -m pytest tests
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements to the penalty logic, UI, or model integration.
