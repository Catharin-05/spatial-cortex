# 👁️ Spatial Cortex — Autonomous Local Video Intelligence

> **Chat with your construction site footage. Zero cloud. Zero cost. Zero data leaks.**

Spatial Cortex is a fully **local, privacy-first AI agent** that lets you ask plain-English questions about any construction video and receive structured audit reports — without sending a single frame to the cloud. It combines a **ReAct reasoning loop** (powered by a locally-hosted LLM via Ollama) with an optimized edge vision pipeline built on **YOLOv8 ONNX** and **OpenCV** to deliver enterprise-grade site intelligence on commodity hardware.

---

## 🎬 Demo

### 1 — Upload & Query
![Agent initializing and searching for trucks across the video timeline](sample/Spatial%20Cortex%20UI%20and%201%20more%20page%20-%20Person%204%20-%20Microsoft%E2%80%8B%20Edge%2018-04-2026%2009_14_07.png)

> The agent receives the query *"Find where the truck is and check if it is a cement truck."*, initialises the agentic flow, and executes `search_objects` to locate truck timestamps across the video feed.

---

### 2 — Multi-Cycle Reasoning & VQA Inspection
![Agent running VQA on a candidate frame and generating the final audit report](sample/Spatial%20Cortex%20UI%20and%201%20more%20page%20-%20Person%204%20-%20Microsoft%E2%80%8B%20Edge%2018-04-2026%2009_15_04.png)

> In Cycle 2 the agent calls `analyze_state` on the most promising timestamp, running LLaVA visual Q&A. In Cycle 3 it concludes and emits the **Site Audit Report**.

---

### 3 — Structured Audit Report
![Final JSON audit report showing the cement truck location at 11 seconds](sample/Spatial%20Cortex%20UI%20and%201%20more%20page%20-%20Person%204%20-%20Microsoft%E2%80%8B%20Edge%2018-04-2026%2009_15_32.png)

> The report is returned as structured JSON — ready for downstream dashboards, ticketing systems, or compliance logs.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🔒 **100 % Local** | Ollama (Llama 3.2 + LLaVA) — no API keys, no data egress |
| ⚡ **Edge-Optimised YOLO** | YOLOv8 Nano exported to ONNX for fast CPU inference |
| 🚦 **Gatekeeper Architecture** | OpenCV motion-diff gates YOLO — up to **90 % less compute** on static feeds |
| 🔍 **Coarse-to-Fine Search** | 1-frame-per-2s coarse pass → fine-grained zoom on hits |
| 🧠 **ReAct Agent Loop** | Iterative Reasoning + Acting cycle (up to 5 retries) with tool memory |
| 🎙️ **Natural Language Queries** | Ask anything — *"Is the driver wearing a helmet?"* |
| 🖥️ **Streamlit UI** | Upload video, type query, watch the agent think in real time |
| 🗂️ **Zero-cost Memory** | Lightweight JSON caching — repeated questions answer instantly |

---

## 🏗️ Architecture

```
User Query (Natural Language)
        │
        ▼
┌─────────────────────┐
│   Ollama LLM Brain  │  ← llama3.2 (local)
│   ReAct Loop        │
└──────────┬──────────┘
           │  selects tool
    ┌──────┴──────────────────────────┐
    │                                 │                        
    ▼                                 ▼                      
search_objects()            analyze_state()         count_unique_objects()    check_progress()
(YOLOv8n ONNX               (LLaVA VQA via          (ByteTrack ID            (First vs Last
 + Motion Gate)              Ollama)                  counting)                Frame diff)
    │                                 │
    └──────────────┬──────────────────┘
                   ▼
          Observation injected
          back into LLM memory
                   │
                   ▼
           Final Audit Report
           (structured JSON)
```

### Engineering Methods

**Method A — The Gatekeeper (Cascading Triggers)**
OpenCV `absdiff` pixel-frame differencing acts as a near-zero-cost motion detector. YOLO only wakes up when pixels actually change — dramatically reducing compute on static camera feeds.

**Method B — Coarse-to-Fine Search**
The agent first scans 1 frame every 2 seconds (coarse). On a detection hit it zooms into that clip at full frame-rate (fine), finding the exact moment an event occurred across hour-long recordings.

**Method C — Neural Fingerprinting / Metadata Caching**
Every search result is cached to a lightweight local JSON dictionary. Repeated queries on the same timeframe are answered instantly from cache without re-running any model.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM Brain** | [Ollama](https://ollama.com/) — `llama3.2` |
| **Vision LLM (VQA)** | Ollama — `llava` |
| **Object Detection** | YOLOv8 Nano (`.onnx`) via Ultralytics |
| **ONNX Runtime** | `onnxruntime` (CPU) |
| **Video Processing** | OpenCV (`opencv-python`) |
| **Object Tracking** | ByteTrack via Ultralytics `track()` |
| **UI** | Streamlit |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
spatial-cortex/
├── app.py                   # Streamlit UI — upload video, run audit
├── agent.py                 # SpatialCortexAgent — ReAct loop & streaming
├── tools.py                 # Legacy standalone detection utilities
├── tools/
│   ├── detector.py          # search_objects() — YOLO object search
│   ├── inspector.py         # analyze_state() — LLaVA visual Q&A
│   ├── tracker.py           # count_unique_objects() — ByteTrack counting
│   └── compare.py           # check_progress() — first/last frame diff
├── helpers/
│   └── optimize_model.py    # One-time script: exports YOLOv8n → ONNX
├── yolov8n.onnx             # Pre-exported edge model (tracked in repo)
├── sample/                  # Demo screenshots
├── test-videos/             # Local test footage (git-ignored)
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running locally

### 1. Clone & Install

```bash
git clone https://github.com/your-username/spatial-cortex.git
cd spatial-cortex
pip install -r requirements.txt
```

### 2. Pull the Required Ollama Models

```bash
ollama pull llama3.2
ollama pull llava
```

### 3. (Optional) Re-export the ONNX Model

The repo ships with a pre-built `yolov8n.onnx`. If you need to regenerate it:

```bash
python helpers/optimize_model.py
```

### 4. Launch the UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload an MP4 construction video, type your audit query, and click **Run Audit**.

---

## 💬 Example Queries

```
"Find where the cement truck is and check if it has finished unloading."
"Count how many unique trucks passed through the site today."
"Check if any worker near the crane is not wearing a hard hat."
"Summarize what changed on site between the beginning and end of this clip."
```

---

## 🔧 Available Agent Tools

| Tool | Signature | Purpose |
|---|---|---|
| `search_objects` | `(target_object: str)` | Returns timestamps where an object class is detected (COCO classes) |
| `analyze_state` | `(timestamp_sec: int, question: str)` | Runs LLaVA visual Q&A on a specific frame |
| `count_unique_objects` | `(target: str)` | Uses ByteTrack to count unique object identities across the full video |
| `check_progress` | `()` | Compares the first and last frames to summarise site progression |

> **Note:** `search_objects` uses standard COCO class names (`person`, `truck`, `car`, etc.). The agent is instructed to resolve higher-level queries (e.g. *"cement truck"*) by first searching for `truck` then calling `analyze_state` for visual disambiguation.

---

## ⚙️ Configuration

All configuration is done at runtime through the Streamlit sidebar:

| Setting | Default | Description |
|---|---|---|
| Video file | — | MP4 upload (up to 200 MB) |
| Audit query | `"Find where the truck is..."` | Natural language instruction for the agent |
| LLM model | `llama3.2` | Configurable in `agent.py` constructor |
| Max retries | `5` | Max ReAct loop cycles before forced exit |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for edge-first AI. No cloud required.
</p>