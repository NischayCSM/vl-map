# VL-MaP: Vision-Language Matrix Penalty Search

**An Intelligent Multimodal Video Analysis System**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![CLIP](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F16-green) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Overview

**VL-MaP** (Vision-Language Matrix Penalty) is an intelligent video analysis tool designed to perform precise, semantic retrieval on educational and technical video content.

Standard vector search tools often struggle with "visual synonyms" in lecture videos—for example, confusing a PowerPoint slide *about* code with an actual *code editor* interface. **VL-MaP** solves this by introducing a novel **Matrix-Penalty Architecture**. It combines a positive search vector (what you want) with a negative penalty vector (what you don't want) to mathematically filter out false positives in real-time.

The system features a **Smart Router** that distinguishes between visual queries ("Find code") and conceptual queries ("Explain recursion"), automatically routing the latter to a Retrieval-Augmented Generation (RAG) engine powered by **Llama 3**.

## ✨ Key Features

* **👁️ Matrix-Penalty Visual Search:**
    * Uses **CLIP (ViT-B/16 Patch16)** for high-resolution visual embedding.
    * Implements a custom linear algebra search engine using direct tensor operations (eliminating the need for a vector database).
    * Applies a dynamic penalty score equation:  
      $$Score = Max(Positive) - (Penalty \times Max(Negative))$$
* **🧠 Intelligent RAG Explainer:**
    * Uses **Faster-Whisper** for high-accuracy speech-to-text transcription.
    * Integrates **Llama 3** (via Ollama) to answer conceptual questions based on the transcript context.
* **🚦 Smart Intent Router:**
    * Automatically detects if a user wants to *see* something (Visual Mode) or *learn* something (Explain Mode) and switches processing paths instantly.
* **🚀 Efficient Architecture:**
    * **Phase 1 (Ingestion):** Pre-computes video matrices and transcripts once per video.
    * **Phase 2 (Inference):** Runs searches in milliseconds using cached tensors loaded in VRAM.
    * **Stack-Based UI:** Results appear chronologically (newest on top) for a seamless search engine experience.

## 🏗️ Architecture

The system operates in two distinct phases:

1.  **Ingestion:** Converts raw video frames and audio into a **Video Matrix** (`.pt` file) and a **Transcript** (`.txt` file).
2.  **Inference:** Performs dot-product similarity search with penalty logic for visuals, or LLM-based RAG for concepts.

## 🛠️ Installation

### Prerequisites
* **Python 3.10+**
* **Ollama** (Must be installed and running locally)
* **FFmpeg** (Required for audio extraction)
* **GPU** (Recommended: CUDA for fast inference)

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/vl-map.git](https://github.com/your-username/vl-map.git)
cd vl-map
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
###Setup Ollama

Download and run Ollama, then pull the Llama 3 model:
```bash
ollama pull llama3
```
#🚀 Usage

###Run the Application

```bash
streamlit run app.py
```
###Analyze a Video

*Open the web interface (default: http://localhost:8501).
*Upload an MP4 video file using the sidebar.
*Click "🚀 Analyze".
The system will process visuals (CLIP) and audio (Whisper). This takes a few minutes depending on video length and GPU power.

###Search & Chat

*Visual Search: Type queries like "Find python code", "Show me the flow chart".
    *The system will return a list of clickable timestamps leading to the exact frame.
*Conceptual Explanation: Type queries like "Explain the algorithm", "Summarize the intro".
    *The system will read the transcript and provide a professor-style explanation.

##Project Structure

vl-map/ ├── app.py # Main Streamlit Application (The "Intelligent System") ├── requirements.txt # Python Dependencies ├── video_data/ # Generated Assets (Gitignored) │ ├── current_video.mp4 │ ├── video_vectors.pt # The Video Matrix (Tensor File) │ └── transcript.txt # Speech-to-Text Output └── README.md # Documentation

##Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements to the penalty logic, UI, or model integration.
