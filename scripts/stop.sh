#!/bin/bash
set -e

echo "========================================="
echo "  Stopping Churn Prediction Services"
echo "========================================="

# Kill by PID files
for pidfile in /tmp/churn_api.pid /tmp/churn_dashboard.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping PID $pid..."
            kill $pid 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

# Kill by port
for port in 5000 8501 8765; do
    pid=$(lsof -t -i:$port 2>/dev/null) || true
    if [ -n "$pid" ]; then
        echo "Stopping process on port $port (PID: $pid)..."
        kill $pid 2>/dev/null || true
    fi
done

# Kill Python processes
pkill -f "src/deployment/api/app.py" 2>/dev/null || true
pkill -f "streamlit run src/deployment/dashboard" 2>/dev/null || true

echo ""
echo "========================================="
echo "  All Services Stopped!"
echo "========================================="