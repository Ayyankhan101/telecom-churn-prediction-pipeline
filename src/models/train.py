import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/ayyan/project/intership/data/processed")
MODEL_DIR = Path("/home/ayyan/project/intership/models")
REPORT_DIR = Path("/home/ayyan/project/intership/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv", usecols=[0]).iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv", usecols=[0]).iloc[:, 0]
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_prob, model_name: str) -> dict:
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }


def tune_hyperparameters(X_train, y_train):
    """Perform hyperparameter tuning for LightGBM"""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
    }
    
    lgb_model = lgb.LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        lgb_model,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Churn rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}\n")
    
    all_results = []
    
    print("="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {
        'LogisticRegression_balanced': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'RandomForest_balanced': RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', 
            random_state=42, n_jobs=-1
        ),
        'XGBoost_scaled': xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            scale_pos_weight=3.5, random_state=42, eval_metric='logloss', n_jobs=-1
        ),
        'LightGBM_unbalance': lgb.LGBMClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1
        ),
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob, name)
        all_results.append(metrics)
        trained_models[name] = model
        
        print(f"  F1: {metrics['f1']:.3f}, Recall: {metrics['recall']:.3f}, AUC: {metrics['roc_auc']:.3f}")
    
    print("\n" + "="*50)
    print("ENSEMBLE MODELS")
    print("="*50)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        scale_pos_weight=3.5, random_state=42, eval_metric='logloss', n_jobs=-1
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1
    )
    
    voting = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft'
    )
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    y_prob = voting.predict_proba(X_test)[:, 1]
    voting_metrics = evaluate_model(y_test, y_pred, y_prob, "VotingEnsemble")
    all_results.append(voting_metrics)
    trained_models['VotingEnsemble'] = voting
    print(f"Voting Ensemble - F1: {voting_metrics['f1']:.3f}, Recall: {voting_metrics['recall']:.3f}")
    
    stacking = StackingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    y_prob = stacking.predict_proba(X_test)[:, 1]
    stacking_metrics = evaluate_model(y_test, y_pred, y_prob, "StackingEnsemble")
    all_results.append(stacking_metrics)
    trained_models['StackingEnsemble'] = stacking
    print(f"Stacking Ensemble - F1: {stacking_metrics['f1']:.3f}, Recall: {stacking_metrics['recall']:.3f}")
    
    results_df = pd.DataFrame(all_results).sort_values('f1', ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    
    best_idx = results_df['f1'].idxmax()
    best_row = results_df.loc[best_idx]
    best_model_name = best_row['model']
    best_model = trained_models[best_model_name]
    best_metrics = best_row.to_dict()
    
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1:        {best_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    
    print("\n" + "="*50)
    print("ALL MODELS RANKED BY F1")
    print("="*50)
    print(results_df.to_string(index=False))
    
    print(f"\nSaved: {MODEL_DIR / 'best_model.joblib'}")