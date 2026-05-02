#!/bin/bash
set -e

echo "========================================="
echo "  Restarting Churn Prediction Stack"
echo "========================================="

# Stop everything
echo "Stopping services..."
docker compose down --remove-orphans

# Fresh start
echo "Building and starting fresh..."
docker compose up -d --build

echo ""
echo "========================================="
echo "  Stack Restarted Successfully!"
echo "  API:       http://localhost:5000"
echo "  Dashboard: http://localhost:8501"
echo "========================================="