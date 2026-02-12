#!/bin/bash
# Start the BillBot Memory Cortex middleware
# This runs in WSL2 and connects to:
#   - llama-server on Windows (localhost:8301, Radeon VII via Vulkan)
#   - SQLite database at ~/.openclaw/memory-cortex/memories.db

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"
LOG_FILE="/tmp/memory-cortex.log"
PID_FILE="/tmp/memory-cortex.pid"

# Check if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Memory Cortex is already running (PID: $(cat "$PID_FILE"))"
    exit 1
fi

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    /usr/bin/python3.12 -m venv --without-pip "$VENV_DIR"
    "$VENV_DIR/bin/python" -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"
    "$VENV_DIR/bin/python" /tmp/get-pip.py
    "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
fi

# Ensure database directory exists
mkdir -p /home/mferr/.openclaw/memory-cortex

echo "Starting Memory Cortex middleware..."
echo "  Log: $LOG_FILE"
echo "  Model endpoint: http://localhost:8301/v1"
echo "  Database: /home/mferr/.openclaw/memory-cortex/memories.db"

cd "$PROJECT_DIR"
nohup "$VENV_DIR/bin/python" -m middleware.server "$PROJECT_DIR/config/config.yaml" \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Memory Cortex started (PID: $(cat "$PID_FILE"))"
