# Harp String Detection — Web Interface

Web UI for:
- **Audio detection**: Colab-trained .keras model + video → onset detection (default or hybrid with YIN fallback).
- **Hand detection**: YOLO string detection + MediaPipe hands → touch events (same as `harp_hand_detector.py`).

## Prerequisites

- **Python 3.10+** with backend dependencies
- **Node.js 18+** for the React frontend
- **FFmpeg** on PATH (for audio pipeline)
- **Hand detection**: put `best.pt` in `backend/weights/` or project root, or upload a .pt file in the UI

## 1. Backend (Python)

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Leave this running. The API will be at `http://127.0.0.1:8000` (docs at `http://127.0.0.1:8000/docs`).

## 2. Frontend (React)

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

## Usage

**Audio detection**
1. Choose **Audio (model + onset detection)**.
2. Select your `.keras` model and a video. Pick **Default** or **Hybrid** (YIN fallback).
3. Run detection → download CSV and labeled video.

**Hand detection**
1. Choose **Hand (YOLO strings + MediaPipe touch)**.
2. Select a video. Optionally upload a `.pt` weights file (or place `best.pt` in `backend/weights/`).
3. Run detection → download touch-events CSV and annotated video.

## For presentation

- Run backend and frontend on your laptop, or deploy the backend to a server and set the frontend API base URL to that server.
- Teachers can use the UI without opening Colab or the command line.
