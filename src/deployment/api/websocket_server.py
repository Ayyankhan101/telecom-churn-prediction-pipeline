import asyncio
import json
import time
import logging
import os
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root to path for running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get('MODEL_PATH', str(Path(__file__).parent.parent.parent.parent / 'models')))

try:
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    encoders = joblib.load(MODEL_DIR / "encoders.joblib")
    logger.info(f"Models loaded successfully from {MODEL_DIR}")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    model = None
    scaler = None
    encoders = None

from src.deployment.api.shared import preprocess_input
from src.deployment.api.utils import RateLimiter, PerformanceTracker

rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
performance_tracker = PerformanceTracker()


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, set] = defaultdict(set)
    
    async def connect(self, websocket, room: str = "global"):
        await websocket.accept()
        self.active_connections[room].add(websocket)
        logger.info(f"Client connected to room: {room}")
    
    def disconnect(self, websocket, room: str = "global"):
        self.active_connections[room].discard(websocket)
    
    async def broadcast(self, message: dict, room: str = "global"):
        if room in self.active_connections:
            await asyncio.gather(
                *[conn.send(json.dumps(message)) for conn in self.active_connections[room]],
                return_exceptions=True
            )


manager = ConnectionManager()


async def handle_client(websocket, path):
    room = "predictions"
    await manager.connect(websocket, room)
    
    try:
        async for message in websocket:
            start_time = time.time()
            
            try:
                data = json.loads(message)
                
                client_id = data.get("client_id", "anonymous")
                
                if not rate_limiter.is_allowed(client_id):
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Rate limit exceeded",
                        "retry_after": 60
                    }))
                    continue
                
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                    continue
                
                if "features" in data:
                    if model is None:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Model not loaded. Check server logs."
                        }))
                        continue
                    X = preprocess_input(data["features"], encoders=encoders, scaler=scaler)
                    prediction = model.predict(X)[0]
                    probability = model.predict_proba(X)[0]
                    
                    latency_ms = (time.time() - start_time) * 1000
                    performance_tracker.record(latency_ms)
                    
                    result = {
                        "type": "prediction",
                        "prediction": "Churn" if prediction == 1 else "No Churn",
                        "churn_probability": float(probability[1]),
                        "no_churn_probability": float(probability[0]),
                        "latency_ms": round(latency_ms, 2),
                        "timestamp": time.time()
                    }
                    
                    await websocket.send(json.dumps(result))
                    
                elif data.get("type") == "batch":
                    if model is None:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Model not loaded"
                        }))
                        continue
                    results = []
                    for features in data.get("features", []):
                        X = preprocess_input(features, encoders=encoders, scaler=scaler)
                        prediction = model.predict(X)[0]
                        probability = model.predict_proba(X)[0]
                        results.append({
                            "prediction": "Churn" if prediction == 1 else "No Churn",
                            "churn_probability": float(probability[1])
                        })
                    
                    await websocket.send(json.dumps({
                        "type": "batch_results",
                        "results": results,
                        "timestamp": time.time()
                    }))
                
                elif data.get("type") == "subscribe":
                    target_room = data.get("room", "predictions")
                    manager.disconnect(websocket, room)
                    manager.active_connections[target_room].add(websocket)
                    room = target_room
                    await websocket.send(json.dumps({
                        "type": "subscribed",
                        "room": target_room
                    }))
                
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Unknown message type"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, room)


async def broadcast_predictions():
    """Background task to broadcast model metrics periodically"""
    while True:
        await asyncio.sleep(30)
        
        elapsed = time.time() - performance_tracker.start_time
        metrics = {
            "type": "metrics",
            "total_predictions": performance_tracker.total_requests,
            "throughput_rps": round(performance_tracker.total_requests / elapsed, 2) if elapsed > 0 else 0,
            "timestamp": time.time()
        }
        
        await manager.broadcast(metrics, "metrics")


async def start_server():
    logger.info("Starting WebSocket server on ws://0.0.0.0:8765")
    
    server = await asyncio.start_server(
        handle_client,
        "0.0.0.0",
        8765
    )
    
    asyncio.create_task(broadcast_predictions())
    
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(start_server())