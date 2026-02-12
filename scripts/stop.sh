#!/bin/bash
# Stop the BillBot Memory Cortex middleware

PID_FILE="/tmp/memory-cortex.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Memory Cortex is not running (no PID file)"
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping Memory Cortex (PID: $PID)..."
    kill "$PID"
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing..."
        kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    echo "Memory Cortex stopped."
else
    echo "Memory Cortex is not running (stale PID: $PID)"
    rm -f "$PID_FILE"
fi
