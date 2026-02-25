import requests
import time
import os

print("Testing Harp String Detection API Locally...")

url = "http://127.0.0.1:8000/api/upload"

# Prepare files for hybrid fast mode
video_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/sample.mp4"
weights_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/best.pt"
model_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/backend/uploads/c9f80e35-b373-4826-943d-bbc231902c96/harp_crnn_freq_025.keras"

if not os.path.exists(model_file):
    print("Error: Could not find uploaded .keras model.")
    exit(1)

files = [
    ("video", ("sample.mp4", open(video_file, "rb"), "video/mp4")),
    ("model", ("model.keras", open(model_file, "rb"), "application/octet-stream")),
    ("weights", ("best.pt", open(weights_file, "rb"), "application/octet-stream"))
]
data = {
    "method": "both",
    "mode": "hybrid"
}

t0 = time.time()
try:
    print("Uploading job...")
    response = requests.post(url, files=files, data=data)
    response.raise_for_status()
    result = response.json()
    job_id = result.get("job_id")
    print(f"Job started: {job_id}")
    
    while True:
        status_resp = requests.get(f"http://127.0.0.1:8000/api/status/{job_id}")
        st_data = status_resp.json()
        print(f"Status: {st_data['status']} - {st_data.get('message', '')}")
        if st_data["status"] == "done":
            print(f"Success! Job took {time.time() - t0:.2f}s")
            print(f"Combined JSON: {st_data.get('combined', {}).get('json_path')}")
            break
        elif st_data["status"] == "error":
            print(f"Failed: {st_data.get('message')}")
            break
        time.sleep(1.0)
except Exception as e:
    print(f"Request failed: {e}")

