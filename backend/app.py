"""
Harp string detection API: audio (model + video) or hand detection (video + optional weights).
Run: python -m uvicorn app:app --reload
"""

import os
import sys
import uuid
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from inference import run_pipeline

# Import hand detector from project root (parent of backend)
_backend_dir = Path(__file__).resolve().parent
_project_root = _backend_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
try:
    from harp_hand_detector import run as run_hand_detector
except ImportError as e:
    import traceback
    print(f"Warning: Could not import harp_hand_detector: {e}")
    print(traceback.format_exc())
    run_hand_detector = None
except Exception as e:
    import traceback
    print(f"Warning: Error loading harp_hand_detector: {e}")
    print(traceback.format_exc())
    run_hand_detector = None

app = FastAPI(title="Harp String Detection API")

# CORS: set ALLOWED_ORIGINS env var for production (comma-separated)
# e.g. ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:5173
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
_origins = [o.strip() for o in _env_origins.split(",") if o.strip()] if _env_origins else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = _backend_dir / "uploads"
OUTPUT_DIR = _backend_dir / "outputs"
WEIGHTS_DIR = _backend_dir / "weights"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# Default YOLO weights for hand detection (user can put best.pt here or upload)
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"
FALLBACK_WEIGHTS = _project_root / "best.pt"

jobs = {}


def run_job_audio(job_id: str, model_path: str, video_path: str, use_yin_fallback: bool):
    try:
        jobs[job_id] = {"status": "running", "message": "Processing (audio)..."}
        csv_path, video_out_path, df = run_pipeline(
            model_path, video_path, str(OUTPUT_DIR / job_id), use_yin_fallback=use_yin_fallback
        )
        jobs[job_id] = {
            "status": "done",
            "csv_path": csv_path,
            "video_path": video_out_path,
            "rows": len(df),
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


def run_job_hand(job_id: str, video_path: str, weights_path: str | None):
    if run_hand_detector is None:
        jobs[job_id] = {"status": "error", "message": "Hand detector not available (harp_hand_detector not found)."}
        return
    try:
        jobs[job_id] = {"status": "running", "message": "Processing (hand detection)..."}
        out_dir = str(OUTPUT_DIR / job_id)
        csv_path, video_out_path = run_hand_detector(
            video_path,
            output_dir=out_dir,
            preview=False,
            weights_path=weights_path,
        )
        rows = 0
        if os.path.isfile(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = max(0, sum(1 for _ in f) - 1)
        jobs[job_id] = {
            "status": "done",
            "csv_path": csv_path,
            "video_path": video_out_path,
            "rows": rows,
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


def run_job_both(
    job_id: str,
    model_path: str,
    video_path: str,
    use_yin_fallback: bool,
    weights_path: str | None,
    original_video_path: str,
):
    out_dir = str(OUTPUT_DIR / job_id)
    try:
        jobs[job_id] = {"status": "running", "message": "Processing (audio)..."}
        csv_audio, video_audio, df = run_pipeline(
            model_path, video_path, out_dir, use_yin_fallback=use_yin_fallback
        )
        audio_result = {"csv_path": csv_audio, "video_path": video_audio, "rows": len(df)}

        if run_hand_detector is None:
            jobs[job_id] = {
                "status": "done",
                "audio": audio_result,
                "hand": None,
                "hand_error": "Hand detector not available (install ultralytics, mediapipe, opencv-python and put best.pt in backend/weights/).",
            }
            return
        jobs[job_id] = {"status": "running", "message": "Processing (hand)..."}
        try:
            csv_hand, video_hand = run_hand_detector(
                video_path, output_dir=out_dir, preview=False, weights_path=weights_path
            )
            hand_rows = 0
            if os.path.isfile(csv_hand):
                with open(csv_hand, "r", encoding="utf-8") as f:
                    hand_rows = max(0, sum(1 for _ in f) - 1)
            hand_result = {"csv_path": csv_hand, "video_path": video_hand, "rows": hand_rows}
            
            # Create combined video: hand annotations + audio subtitles
            jobs[job_id] = {"status": "running", "message": "Combining results..."}
            srt_path = os.path.join(out_dir, "overlay.srt")
            combined_video = os.path.join(out_dir, "video_combined.mp4")
            combined_result = None
            combined_error = None
            
            if not os.path.isfile(srt_path):
                combined_error = f"SRT file not found: {srt_path}"
            elif not os.path.isfile(video_hand):
                combined_error = f"Hand video not found: {video_hand}"
            else:
                # Burn SRT subtitles onto hand-detected video, use original video for audio
                from inference import FFMPEG_CMD
                srt_abs = os.path.abspath(srt_path).replace("\\", "/")
                if os.name == "nt":
                    srt_abs = srt_abs.replace(":", "\\:", 1)
                try:
                    # Use hand video for video stream, original video for audio stream, burn SRT
                    result = subprocess.run([
                        FFMPEG_CMD, "-y",
                        "-i", video_hand,  # Video with hand annotations
                        "-i", original_video_path,  # Original video for audio
                        "-vf", f"subtitles='{srt_abs}':force_style='Fontsize=36,BorderStyle=1,Outline=2,Shadow=1,MarginV=50'",
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-map", "0:v:0",  # Video from first input (hand video)
                        "-map", "1:a:0",  # Audio from second input (original video)
                        "-shortest",  # End when shortest stream ends
                        combined_video
                    ], check=True, capture_output=True, text=True)
                    if os.path.isfile(combined_video):
                        combined_result = {"video_path": combined_video}
                    else:
                        combined_error = "Combined video file was not created"
                except subprocess.CalledProcessError as e:
                    combined_error = f"FFmpeg error: {e.stderr[:200] if e.stderr else str(e)}"
                except Exception as e:
                    combined_error = f"Error creating combined video: {str(e)}"
            
            jobs[job_id] = {
                "status": "done",
                "audio": audio_result,
                "hand": hand_result,
                "combined": combined_result,
                "combined_error": combined_error,
            }
        except Exception as hand_err:
            jobs[job_id] = {
                "status": "done",
                "audio": audio_result,
                "hand": None,
                "hand_error": str(hand_err),
            }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


@app.post("/api/upload")
async def upload_and_run(
    background_tasks: BackgroundTasks,
    method: str = Form("audio"),
    model: UploadFile = File(None),
    video: UploadFile = File(...),
    mode: str = Form("hybrid"),
    weights: UploadFile = File(None),
):
    if not video.filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        raise HTTPException(400, "Video must be .mp4, .mov, .mkv, .avi, or .webm")

    method = method.lower()
    if method not in ("audio", "hand", "both"):
        raise HTTPException(400, "method must be 'audio', 'hand', or 'both'")

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)
    (OUTPUT_DIR / job_id).mkdir(parents=True, exist_ok=True)

    video_path = job_dir / (video.filename or "video.mp4")
    try:
        with open(video_path, "wb") as f:
            f.write(await video.read())
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Save video failed: {e}")

    if method == "audio":
        if not model or not model.filename or not model.filename.lower().endswith(".keras"):
            raise HTTPException(400, "For audio detection, upload a .keras model file.")
        model_path = job_dir / (model.filename or "model.keras")
        try:
            with open(model_path, "wb") as f:
                f.write(await model.read())
        except Exception as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(500, f"Save model failed: {e}")
        use_yin = mode.lower() == "hybrid"
        jobs[job_id] = {"status": "queued"}
        background_tasks.add_task(run_job_audio, job_id, str(model_path), str(video_path), use_yin)
    elif method == "both":
        if not model or not model.filename or not model.filename.lower().endswith(".keras"):
            raise HTTPException(400, "For both, upload a .keras model file.")
        model_path = job_dir / (model.filename or "model.keras")
        try:
            with open(model_path, "wb") as f:
                f.write(await model.read())
        except Exception as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(500, f"Save model failed: {e}")
        use_yin = mode.lower() == "hybrid"
        weights_path = None
        if weights and weights.filename and weights.filename.lower().endswith(".pt"):
            wpath = job_dir / (weights.filename or "weights.pt")
            try:
                with open(wpath, "wb") as f:
                    f.write(await weights.read())
            except Exception as e:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(500, f"Save weights failed: {e}")
            weights_path = str(wpath)
        else:
            if DEFAULT_WEIGHTS.exists():
                weights_path = str(DEFAULT_WEIGHTS)
            elif FALLBACK_WEIGHTS.exists():
                weights_path = str(FALLBACK_WEIGHTS)
        jobs[job_id] = {"status": "queued"}
        background_tasks.add_task(
            run_job_both, job_id, str(model_path), str(video_path), use_yin, weights_path, str(video_path)
        )
    else:
        weights_path = None
        if weights and weights.filename and weights.filename.lower().endswith(".pt"):
            weights_path = job_dir / (weights.filename or "weights.pt")
            try:
                with open(weights_path, "wb") as f:
                    f.write(await weights.read())
            except Exception as e:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(500, f"Save weights failed: {e}")
            weights_path = str(weights_path)
        else:
            if DEFAULT_WEIGHTS.exists():
                weights_path = str(DEFAULT_WEIGHTS)
            elif FALLBACK_WEIGHTS.exists():
                weights_path = str(FALLBACK_WEIGHTS)
            else:
                raise HTTPException(
                    400,
                    "For hand detection, upload a .pt weights file or place best.pt in backend/weights/ or project root.",
                )
        jobs[job_id] = {"status": "queued"}
        background_tasks.add_task(run_job_hand, job_id, str(video_path), weights_path)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


def _get_result_path(job: dict, kind: str, type_key: str | None) -> tuple[str, str]:
    """Return (path, filename). type_key is 'audio', 'hand', or 'combined' for both-mode jobs."""
    if "audio" in job:
        if type_key == "combined":
            combined = job.get("combined")
            if not combined:
                raise HTTPException(404, "No combined video available")
            path = combined.get("video_path")
            name = "harp_combined.mp4"
        elif not type_key:
            raise HTTPException(400, "For this job, use ?type=audio, ?type=hand, or ?type=combined")
        else:
            part = job.get(type_key)
            if not part:
                raise HTTPException(404, f"No {type_key} result")
            path = part.get("csv_path" if kind == "csv" else "video_path")
            name = f"harp_{type_key}_{'predictions' if kind == 'csv' else 'video'}.{'csv' if kind == 'csv' else 'mp4'}"
    else:
        path = job.get("csv_path" if kind == "csv" else "video_path")
        name = "harp_predictions.csv" if kind == "csv" else "harp_labeled.mp4"
    return (path, name)


@app.get("/api/download/csv/{job_id}")
def download_csv(job_id: str, type: str = Query(None, alias="type")):
    if job_id not in jobs or jobs[job_id].get("status") != "done":
        raise HTTPException(404, "Job not ready or not found")
    job = jobs[job_id]
    path, filename = _get_result_path(job, "csv", type)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "CSV not found")
    return FileResponse(path, filename=filename, media_type="text/csv")


@app.get("/api/download/video/{job_id}")
def download_video(job_id: str, type: str = Query(None, alias="type")):
    if job_id not in jobs or jobs[job_id].get("status") != "done":
        raise HTTPException(404, "Job not ready or not found")
    job = jobs[job_id]
    path, filename = _get_result_path(job, "video", type)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, filename=filename, media_type="video/mp4")


@app.get("/")
def root():
    return {"message": "Harp String Detection API", "docs": "/docs"}
