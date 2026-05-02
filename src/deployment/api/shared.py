import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path for running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.feature_engineering import engineer_features
from src.deployment.api.utils import FEATURE_ORDER

logger = logging.getLogger(__name__)


def preprocess_input(data, encoders=None, scaler=None):
    """Preprocess customer data for prediction."""
    df = pd.DataFrame([data])
    
    df = engineer_features(df)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if encoders and col in encoders:
            try:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            except Exception as e:
                logger.warning(f"Encoder error for {col}: {e}, using fallback")
                unique_vals = list(df[col].unique())
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)
        else:
            try:
                unique_vals = list(df[col].unique())
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)
            except:
                df[col] = 0
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if scaler is not None and hasattr(scaler, 'transform') and numerical_cols:
        try:
            # Align with features seen during fit
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = [c for f in scaler.feature_names_in_ if (c := f) in df.columns]
                df[scaler_features] = scaler.transform(df[scaler_features])
            else:
                df[numerical_cols] = scaler.transform(df[numerical_cols])
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}, skipping")
    
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    
    return df[FEATURE_ORDER]