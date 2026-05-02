import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import sys

# Get project root relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.append(str(PROJECT_ROOT / "src"))
from features.feature_engineering import engineer_features

DATA_PATH = PROJECT_ROOT / "data/raw/synthetic_telco_churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
MODEL_DIR = PROJECT_ROOT / "models"

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records")
    return df

def preprocess(df: pd.DataFrame, fit: bool = True, scaler=None, encoders=None):
    df = df.copy()
    
    df = engineer_features(df)
    
    if fit:
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = SimpleImputer(strategy='median').fit_transform(df[[col]])
    else:
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = scaler.imputation_.transform(df[[col]]) if hasattr(scaler, 'imputation_') else df[col]
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isnull().any():
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
    
    customer_ids = df['customerID']
    target = df['Churn'].map({'Yes': 1, 'No': 0})
    
    features_to_drop = ['customerID', 'Churn']
    df = df.drop(columns=features_to_drop, errors='ignore')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    if fit:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        scaler.numerical_cols = numerical_cols
        scaler.categorical_cols = categorical_cols
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df, target, customer_ids, scaler, encoders

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def save_preprocessors(scaler, encoders, path: Path = MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path / "scaler.joblib")
    joblib.dump(encoders, path / "encoders.joblib")
    print(f"Saved preprocessors to {path}")

def load_preprocessors(path: Path = MODEL_DIR):
    scaler = joblib.load(path / "scaler.joblib")
    encoders = joblib.load(path / "encoders.joblib")
    return scaler, encoders


def create_pipeline(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Create sklearn preprocessing pipeline"""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    X, y, customer_ids, scaler, encoders = preprocess(df, fit=True)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    ids_train, ids_test = train_test_split(
        customer_ids,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    save_preprocessors(scaler, encoders)
    
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    ids_test.to_csv(OUTPUT_DIR / "test_customer_ids.csv", index=False)
    
    print(f"\nTrain: {len(X_train)} samples, Churn rate: {y_train.mean():.3f}")
    print(f"Test: {len(X_test)} samples, Churn rate: {y_test.mean():.3f}")
    print(f"Features: {X.shape[1]}")