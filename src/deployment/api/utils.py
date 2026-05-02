import time
from collections import deque

FEATURE_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
    'TotalCharges', 'NumSupportCalls', 'AvgMonthlyCharge', 'tenure_group',
    'avg_charge_per_month', 'charge_variance', 'charge_increase_ratio',
    'contract_type_encoded', 'num_additional_services', 'has_fiber',
    'is_month_to_month', 'high_risk_combo', 'tenure_contract_ratio',
    'high_support_calls', 'support_call_binned', 'is_electronic_check', 'has_family'
]


class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window_seconds
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True


class PerformanceTracker:
    def __init__(self):
        self.latencies = deque(maxlen=1000)
        self.predictions = deque(maxlen=1000)
        self.errors = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def record(self, latency_ms: float, prediction: int = None, error: bool = False):
        self.latencies.append(latency_ms)
        self.total_requests += 1
        if prediction is not None:
            self.predictions.append(prediction)
        if error:
            self.errors += 1
    
    def get_metrics(self) -> dict:
        latencies = sorted(self.latencies)
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        elapsed = time.time() - self.start_time
        
        return {
            "total_requests": self.total_requests,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p50_latency_ms": round(latencies[p50_idx], 2) if latencies else 0,
            "p95_latency_ms": round(latencies[p95_idx], 2) if latencies else 0,
            "p99_latency_ms": round(latencies[p99_idx], 2) if latencies else 0,
            "throughput_rps": round(self.total_requests / elapsed, 2) if elapsed > 0 else 0,
            "error_rate": round(self.errors / self.total_requests, 4) if self.total_requests > 0 else 0,
            "churn_rate_30d": round(sum(self.predictions) / len(self.predictions), 4) if self.predictions else 0
        }