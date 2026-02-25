import time
import os
import sys

# Load inference functions directly
from inference import run_pipeline
from app import run_job_both, jobs

print("Testing Harp Hybrid Pipeline directly...")

video_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/sample.mp4"
weights_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/best.pt"
model_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/backend/uploads/c9f80e35-b373-4826-943d-bbc231902c96/harp_crnn_freq_025.keras"

# Check definitions
if not os.path.exists(model_file):
    print("WARNING: Using fallback search for model file...")
    import glob
    models = glob.glob("/Users/apple/Downloads/HarpHand (2)/HarpHand/backend/uploads/**/*.keras", recursive=True)
    if not models:
        print("ERROR: Model not found")
        sys.exit(1)
    model_file = models[0]
    print(f"Found model: {model_file}")

job_id = "test_job_123"

# Mock output directory setup from app.py
from app import OUTPUT_DIR, UPLOAD_DIR
(OUTPUT_DIR / job_id).mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("Running hybrid job...")
try:
    # run_job_both is synchronous inside the background task context, so we can just call it
    run_job_both(
        job_id=job_id,
        model_path=model_file,
        video_path=video_file,
        use_yin_fallback=True,  # Hybrid
        weights_path=weights_file,
        original_video_path=video_file
    )
    
    print(f"Job took {time.time() - t0:.2f}s")
    test_job = jobs.get(job_id, {})
    print(f"Result Status: {test_job.get('status')}")
    if test_job.get("status") == "error":
        print(f"Error Message: {test_job.get('message')}")
    else:
        combined = test_job.get('combined', {})
        print(f"Combined JSON path: {combined.get('json_path')}")
        
except Exception as e:
    print(f"Failed to run pipeline: {e}")
    import traceback
    traceback.print_exc()
