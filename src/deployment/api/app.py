from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
import os
import json
import sys
from functools import wraps

# Add project root to path for running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.deployment.api.utils import RateLimiter, PerformanceTracker, FEATURE_ORDER
from src.deployment.api.shared import preprocess_input

# Configure structured logging
LOG_FORMAT = os.environ.get('LOG_FORMAT', 'text')
if LOG_FORMAT == 'json':
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                'timestamp': self.formatTime(record),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module
            })
    logging.basicConfig(level=logging.INFO, formatter=JSONFormatter())
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


class CustomerFeatures(BaseModel):
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., pattern="^(Fiber optic|DSL|No)$")
    OnlineSecurity: str = Field(..., pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No|No internet service)$")
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(..., pattern="^(Electronic check|Mailed check|Bank transfer \\(automatic\\)|Credit card \\(automatic\\))$")
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)
    NumSupportCalls: int = Field(..., ge=0, le=20)
    AvgMonthlyCharge: float = Field(..., gt=0)

class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    no_churn_probability: float

app = Flask(__name__)

# Smart path detection
if os.environ.get('MODEL_PATH'):
    MODEL_DIR = Path(os.environ.get('MODEL_PATH'))
else:
    # Local fallback: find models dir relative to this file
    MODEL_DIR = Path(__file__).resolve().parents[3] / "models"

logger.info(f"Model directory set to: {MODEL_DIR}")

# Load version metadata
try:
    version_file = MODEL_DIR / "version.json"
    if version_file.exists():
        import json
        with open(version_file) as f:
            version_data = json.load(f)
        MODEL_VERSION = version_data.get("version", "1.0.0")
        MODEL_METADATA = {
            "model_type": version_data.get("model_type", "LogisticRegression"),
            "training_date": version_data.get("training_date", "unknown"),
            "features_count": version_data.get("features", 35),
            "target": "Churn"
        }
    else:
        MODEL_VERSION = "1.0.0"
        MODEL_METADATA = {"model_type": "LogisticRegression", "features_count": 35, "target": "Churn"}
except Exception as e:
    logging.warning(f"Version file error: {e}")
    MODEL_VERSION = "1.0.0"
    MODEL_METADATA = {"model_type": "LogisticRegression", "features_count": 35, "target": "Churn"}

try:
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    encoders = joblib.load(MODEL_DIR / "encoders.joblib")
    logger.info(f"Models loaded from {MODEL_DIR}")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    model = scaler = encoders = None

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
performance_tracker = PerformanceTracker()

API_KEY = os.environ.get('API_KEY', 'dev-key-12345')

def verify_api_key():
    """Verify API key from header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return False
    return auth_header.replace('Bearer ', '') == API_KEY

def require_auth(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not verify_api_key():
            return jsonify({'error': 'Unauthorized', 'message': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return wrapped

def rate_limit(limit=100, window=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_id = request.headers.get('X-Client-ID') or request.remote_addr
            if not rate_limiter.is_allowed(client_id):
                return jsonify({'error': 'Rate limit exceeded', 'retry_after': window}), 429
            return f(*args, **kwargs)
        return wrapped
    return decorator


def _preprocess_input(data):
    return preprocess_input(data, encoders=encoders, scaler=scaler)


@app.route('/predict', methods=['POST'])
@rate_limit(limit=100, window=60)
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        
        # Validate input with Pydantic
        try:
            validated_data = CustomerFeatures(**data)
        except Exception as ve:
            logger.error(f"Validation failed: {ve}")
            return jsonify({
                'error': 'Validation failed', 
                'details': str(ve),
                'received_data': data
            }), 422
        
        X = _preprocess_input(validated_data.model_dump())
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        latency_ms = (time.time() - start_time) * 1000
        performance_tracker.record(latency_ms, prediction, error=False)
        
        result = {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'churn_probability': float(probability[1]),
            'no_churn_probability': float(probability[0])
        }
        
        return jsonify(result)
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        performance_tracker.record(latency_ms, error=True)
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/model/info', methods=['GET'])
def model_info():
    metrics = performance_tracker.get_metrics()
    return jsonify({
        'version': MODEL_VERSION,
        'metadata': MODEL_METADATA,
        'performance': metrics
    })


@app.route('/model/ready', methods=['GET'])
def ready():
    return jsonify({
        'ready': model is not None,
        'model_loaded': True
    })


@app.route('/model/feature-importance', methods=['GET'])
def feature_importance():
    """Return top 5 feature importance values"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Check for coef_ (Logistic) or feature_importances_ (Trees)
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return jsonify({'error': 'Model type does not support importance'}), 400

        feat_imp = dict(zip(FEATURE_ORDER, importance))
        sorted_features = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        return jsonify({
            'top_features': [f[0] for f in sorted_features],
            'importance_values': [float(f[1]) for f in sorted_features]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    p = performance_tracker.get_metrics()
    return jsonify({
        'churn_predictions_total': p['total_requests'],
        'churn_prediction_latency_ms_avg': p['avg_latency_ms'],
        'churn_prediction_latency_ms_p95': p['p95_latency_ms'],
        'churn_prediction_latency_ms_p99': p['p99_latency_ms'],
        'churn_prediction_throughput_rps': p['throughput_rps'],
        'churn_prediction_error_rate': p['error_rate'],
        'churn_rate_30d': p['churn_rate_30d']
    })


@app.route('/predict/batch', methods=['POST'])
@rate_limit(limit=20, window=60)
def predict_batch():
    start_time = time.time()
    try:
        data = request.get_json()
        customers = data.get('customers', [])
        
        if not customers:
            return jsonify({'error': 'No customers provided'}), 400
        
        if len(customers) > 100:
            return jsonify({'error': 'Maximum 100 customers per batch'}), 400
        
        results = []
        for customer_data in customers:
            try:
                validated = CustomerFeatures(**customer_data)
                X = preprocess_input(validated.model_dump())
                prediction = model.predict(X)[0]
                probability = model.predict_proba(X)[0]
                results.append({
                    'prediction': 'Churn' if prediction == 1 else 'No Churn',
                    'churn_probability': float(probability[1])
                })
            except Exception as ve:
                results.append({'error': str(ve), 'prediction': 'error'})
        
        latency_ms = (time.time() - start_time) * 1000
        performance_tracker.record(latency_ms, error=False)
        
        return jsonify({
            'results': results,
            'count': len(results),
            'processing_time_ms': round(latency_ms, 2)
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/metrics/prometheus', methods=['GET'])
def prometheus_metrics():
    p = performance_tracker.get_metrics()
    lines = [
        f"# HELP churn_predictions_total Total prediction requests",
        f"# TYPE churn_predictions_total counter",
        f"churn_predictions_total {p['total_requests']}",
        f"# HELP churn_prediction_latency_ms Prediction latency in milliseconds",
        f"# TYPE churn_prediction_latency_ms summary",
        f'churn_prediction_latency_ms_sum {p["avg_latency_ms"] * p["total_requests"]}',
        f'churn_prediction_latency_ms_count {p["total_requests"]}',
        f"# HELP churn_prediction_error_rate Error rate of predictions",
        f"# TYPE churn_prediction_error_rate gauge",
        f"churn_prediction_error_rate {p['error_rate']}",
        f"# HELP churn_rate_30d Churn rate in last 30 days",
        f"# TYPE churn_rate_30d gauge",
        f"churn_rate_30d {p['churn_rate_30d']}"
    ]
    return '\n'.join(lines), 200, {'Content-Type': 'text/plain'}


if __name__ == '__main__':
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)