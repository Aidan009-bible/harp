import requests
import time
import os

AWS_URL = "http://ec2-44-222-62-81.compute-1.amazonaws.com:8000"

print(f"Testing Harp String Detection API directly on AWS: {AWS_URL}")

# Check if the server is up
try:
    health = requests.get(f"{AWS_URL}/")
    print(f"Server health check: {health.json()}")
except Exception as e:
    print(f"Could not connect to AWS server: {e}")
    print("Please ensure your EC2 Security Group allows inbound TCP traffic on port 8000 from your IP address.")
    exit(1)

# Prepare files for hybrid fast mode
video_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/sample.mp4"
weights_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/best.pt"
model_file = "/Users/apple/Downloads/HarpHand (2)/HarpHand/backend/uploads/c9f80e35-b373-4826-943d-bbc231902c96/harp_crnn_freq_025.keras"

if not os.path.exists(model_file):
    import glob
    models = glob.glob("/Users/apple/Downloads/HarpHand (2)/HarpHand/backend/uploads/**/*.keras", recursive=True)
    if models:
        model_file = models[0]
    else:
        print("Error: Could not find any .keras model locally to upload.")
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
    print("Uploading job to AWS...")
    response = requests.post(f"{AWS_URL}/api/upload", files=files, data=data)
    response.raise_for_status()
    result = response.json()
    job_id = result.get("job_id")
    print(f"Upload complete! Job started with ID: {job_id}")
    
    while True:
        status_resp = requests.get(f"{AWS_URL}/api/status/{job_id}")
        st_data = status_resp.json()
        print(f"Status: {st_data['status']} - {st_data.get('message', '')}")
        
        if st_data["status"] == "done":
            print(f"Success! Job took {time.time() - t0:.2f}s")
            # Fetch the actual JSON
            json_resp = requests.get(f"{AWS_URL}/api/download/json/{job_id}?type=combined")
            if json_resp.ok:
                print("Successfully downloaded combined JSON output from AWS!")
                print(f"Preview: {str(json_resp.json())[:200]}...")
            else:
                print("Failed to download JSON.")
            break
        elif st_data["status"] == "error":
            print(f"Failed: {st_data.get('message')}")
            break
            
        time.sleep(2.0)
except Exception as e:
    print(f"Request failed: {e}")
