#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# HarpHand — AWS EC2 (Ubuntu 22.04) Setup Script
# Run:  chmod +x deploy/setup_ec2.sh && sudo bash deploy/setup_ec2.sh
# ────────────────────────────────────────────────────────────────
set -euo pipefail

echo "══════════════════════════════════════════"
echo "  HarpHand Backend — EC2 Setup"
echo "══════════════════════════════════════════"

# ── 1. System packages ──
echo "[1/5] Installing system dependencies..."
apt-get update -y
apt-get install -y python3 python3-pip python3-venv ffmpeg git curl

# ── 2. Project directory ──
PROJECT_DIR="/home/ubuntu/HarpHand"
echo "[2/5] Setting up project in $PROJECT_DIR..."

if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project directory not found at $PROJECT_DIR"
  echo "Please upload the HarpHand project to /home/ubuntu/HarpHand first."
  echo ""
  echo "From your local machine run:"
  echo "  scp -i your-key.pem -r ./HarpHand ubuntu@<EC2-IP>:/home/ubuntu/HarpHand"
  exit 1
fi

# ── 3. Python virtual environment & deps ──
echo "[3/5] Setting up Python environment..."
cd "$PROJECT_DIR/backend"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Also install the root-level requirements (hand detector deps)
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  pip install -r "$PROJECT_DIR/requirements.txt"
fi

deactivate

# ── 4. Systemd service ──
echo "[4/5] Creating systemd service..."

cat > /etc/systemd/system/harphand.service << 'EOF'
[Unit]
Description=HarpHand Backend API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/HarpHand/backend
Environment="PATH=/home/ubuntu/HarpHand/backend/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="ALLOWED_ORIGINS=*"
ExecStart=/home/ubuntu/HarpHand/backend/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable harphand
systemctl start harphand

# ── 5. Verify ──
echo "[5/5] Verifying..."
sleep 2
if systemctl is-active --quiet harphand; then
  echo ""
  echo "══════════════════════════════════════════"
  echo "  ✅ HarpHand backend is running!"
  echo "  API:  http://$(curl -s ifconfig.me):8000"
  echo "  Docs: http://$(curl -s ifconfig.me):8000/docs"
  echo ""
  echo "  NEXT STEPS:"
  echo "  1. Set VITE_API_URL in Vercel to:"
  echo "     http://$(curl -s ifconfig.me):8000/api"
  echo ""
  echo "  2. Make sure port 8000 is open in your"
  echo "     EC2 Security Group."
  echo "══════════════════════════════════════════"
else
  echo "❌ Service failed to start. Check logs with:"
  echo "   sudo journalctl -u harphand -f"
fi
