#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "  Churn Prediction Pipeline Starter"
echo "========================================="

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Checking dependencies..."
pip install -q pydantic websockets prometheus-client 2>/dev/null || true

# Smart MODEL_PATH detection - Docker vs Local
PROJECT_DIR=$(pwd)
if [ -d "/app/models" ]; then
    export MODEL_PATH=/app/models
    echo "  Detected: Docker environment"
else
    # For local, need absolute path
    export MODEL_PATH="${PROJECT_DIR}/models"
    echo "  Detected: Local environment - using ${PROJECT_DIR}/models"
fi

# Export variables
export API_KEY=${API_KEY:-dev-key-12345}
export API_URL=${API_URL:-http://localhost:5000}
export FLASK_DEBUG=false

# Kill existing processes on ports
echo "Checking for existing processes..."
for port in 5000 8501 8765; do
    pid=$(lsof -t -i:$port 2>/dev/null) || true
    if [ -n "$pid" ]; then
        echo "  Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
    fi
done

# Also kill any stale Python processes
pkill -9 -f "src/deployment/api/app.py" 2>/dev/null || true
pkill -9 -f "streamlit run src/deployment/dashboard" 2>/dev/null || true
sleep 2

# Start API
echo "Starting Flask API on port 5000..."
python src/deployment/api/app.py &
API_PID=$!
echo "  API PID: $API_PID"

# Wait for API to be ready
echo "Waiting for API..."
API_READY=false
for i in {1..15}; do
    if curl -s http://localhost:5000/health >/dev/null 2>&1; then
        # Also verify model info endpoint works
        if curl -s http://localhost:5000/model/info >/dev/null 2>&1; then
            echo "  API ready!"
            API_READY=true
            break
        fi
    fi
    sleep 1
done

if [ "$API_READY" = "false" ]; then
    echo "  WARNING: API started but not responding properly"
fi

# Start Dashboard
echo "Starting Streamlit Dashboard on port 8501..."
streamlit run src/deployment/dashboard/app.py --server.port=8501 &
DASH_PID=$!
echo "  Dashboard PID: $DASH_PID"

# Save PIDs
echo $API_PID > /tmp/churn_api.pid
echo $DASH_PID > /tmp/churn_dashboard.pid

echo ""
echo "========================================="
echo "  Services Started Successfully!"
echo "========================================="
echo "  API:      http://localhost:5000"
echo "  Dashboard: http://localhost:8501"
echo ""
echo "To stop: ./scripts/stop.sh"
echo "========================================="