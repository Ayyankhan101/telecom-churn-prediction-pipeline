# Telecom Customer Churn Prediction Pipeline

Machine learning pipeline to predict customer churn for a telecom company.

## Project Overview

**Objective**: Predict whether a customer will churn (leave the company) based on usage, billing, and service data.

**Dataset**: 7,000 synthetic telecom customer records with ~24% churn rate (imbalanced)

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| 1 | **Data Generation** - Synthetic telecom churn data with realistic distributions |
| 2 | **Preprocessing** - Missing values, encoding, scaling, stratified train/test split |
| 3 | **EDA** - Churn distribution, contract analysis, tenure patterns, correlation heatmap |
| 4 | **Feature Engineering** - Tenure groups, usage ratios, risk scores |
| 5 | **Model Training** - LR, RF, XGBoost, LightGBM, Ensembles + hyperparameter tuning |
| 6 | **Evaluation** - Confusion matrix, ROC curve, SHAP interpretability |
| 7 | **Deployment** - Flask API + Streamlit dashboard (Docker-ready) |

## Results

| Model | F1 Score | Recall | ROC-AUC |
|-------|----------|--------|---------|
| Logistic Regression (balanced) | **0.465** | **0.697** | 0.695 |
| Random Forest (balanced) | 0.421 | 0.438 | 0.700 |
| LightGBM (unbalanced) | 0.420 | 0.500 | 0.676 |
| XGBoost (weighted) | 0.391 | 0.465 | 0.667 |

**Best Model**: Logistic Regression with class_weight='balanced'

### Top 5 Churn Drivers (SHAP)
1. **Contract Type** - Month-to-month churns 2x more than two-year
2. **Tenure** - New customers (<12 months) have highest churn
3. **Fiber Optic** - Higher churn despite premium pricing
4. **Monthly Charges** - Higher charges correlate with churn
5. **Electronic Check** - Riskier payment method

## Project Structure

```
/home/ayyan/project/intership/
├── data/
│   ├── raw/                  # Raw/synthetic data
│   └── processed/            # Preprocessed train/test
├── src/
│   ├── data/                 # Data generation
│   ├── preprocessing/        # Preprocessing pipeline (sklearn Pipeline)
│   ├── eda/                  # EDA visualizations
│   ├── features/             # Feature engineering
│   ├── models/               # Training & evaluation
│   └── deployment/           # Flask API + Streamlit
├── models/                   # Saved models (scaler, encoders, best_model)
├── reports/                  # Figures & insights
├── tests/                    # Unit tests
├── .github/workflows/        # CI/CD pipeline
├── Dockerfile                # Docker container definition
├── docker-compose.yml        # Multi-service orchestration
├── venv/                     # Python virtual environment
└── requirements.txt
```

## Quick Start

### Option 1: Local Development

```bash
cd /home/ayyan/project/intership
source venv/bin/activate
pip install -r requirements.txt
```

**Run the pipeline:**

```bash
# 1. Preprocess data
python src/preprocessing/preprocess.py

# 2. Train models (set ENABLE_TUNING=true for hyperparameter tuning)
python src/models/train.py

# 3. Evaluate + SHAP
python src/models/evaluate.py
```

**Start services:**

```bash
# Terminal 1: Flask API
export API_KEY=dev-key-12345
export MODEL_PATH=/home/ayyan/project/intership/models
python src/deployment/api/app.py

# Terminal 2: Streamlit Dashboard
export API_URL=http://localhost:5000
export API_KEY=dev-key-12345
streamlit run src/deployment/dashboard/app.py
```

### Option 2: Docker

```bash
# Build and run all services
docker-compose up --build

# Or rebuild if needed
docker-compose down
docker-compose up --build
```

**Services:**
| Service | URL |
|---------|-----|
| API | http://localhost:5000 |
| WebSocket | ws://localhost:8765 |
| Dashboard | http://localhost:8501 |

### Option 3: Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | ❌ | Health check |
| `POST /predict` | ✅ | Single prediction |
| `POST /predict/batch` | ✅ | Batch prediction (max 100) |
| `GET /model/info` | ✅ | Model info + performance metrics |
| `GET /model/ready` | ✅ | Readiness probe |
| `GET /metrics` | ✅ | JSON metrics |
| `GET /metrics/prometheus` | ✅ | Prometheus format |

### Example API Calls

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.00, "TotalCharges": 840.00,
    "NumSupportCalls": 2, "AvgMonthlyCharge": 65.00
  }'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"customers": [...]}'

# Health check (no auth)
curl http://localhost:5000/health
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `dev-key-12345` | API authentication key |
| `MODEL_PATH` | `/app/models` | Path to model files |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `ENABLE_TUNING` | `false` | Enable hyperparameter tuning |
| `API_URL` | `http://localhost:5000` | Dashboard API URL |

## New Features

- ✅ **API Authentication** - Bearer token auth on protected endpoints
- ✅ **Input Validation** - Pydantic schema validation for all inputs
- ✅ **Rate Limiting** - 100 requests/minute per client
- ✅ **Performance Metrics** - Latency, throughput, error rate tracking
- ✅ **Prometheus Metrics** - `/metrics/prometheus` endpoint
- ✅ **Batch Predictions** - `/predict/batch` endpoint (max 100)
- ✅ **WebSocket Server** - Real-time predictions on port 8765
- ✅ **Docker Support** - Containerized deployment
- ✅ **CI/CD Pipeline** - GitHub Actions workflow
- ✅ **Unit Tests** - pytest test suite

## Business Insights

- **Contract upgrades**: Month-to-month customers churn at ~32% vs ~15% for two-year. Push upgrades.
- **New customer retention**: First 12 months critical (34% churn). Focus retention on new customers.
- **Payment methods**: Electronic check users churn more (28%). Incentivize auto-pay.
- **Fiber optic quality**: High churn (29%) despite premium - investigate service issues.
- **Support intervention**: Track customers with 3+ support calls - high churn risk.

## Dependencies

- pandas, numpy, scikit-learn
- xgboost, lightgbm, imbalanced-learn
- shap, matplotlib, seaborn
- flask, streamlit, joblib
- pydantic, websockets, prometheus-client

## License

For educational purposes - Intern Project 2026