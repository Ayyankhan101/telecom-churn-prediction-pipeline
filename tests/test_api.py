import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import engineer_features
import pandas as pd
import numpy as np


class TestFeatureEngineering:
    def test_tenure_group_creation(self):
        df = pd.DataFrame({'tenure': [6, 24, 36, 60]})
        result = engineer_features(df)
        assert 'tenure_group' in result.columns
        assert result['tenure_group'].iloc[0] == '0-12'
        assert result['tenure_group'].iloc[1] == '12-24'
    
    def test_missing_column_handling(self):
        df = pd.DataFrame({'tenure': [10]})
        result = engineer_features(df)
        assert 'tenure_group' in result.columns
    
    def test_contract_encoding(self):
        df = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'tenure': [10, 10, 10]
        })
        result = engineer_features(df)
        assert 'contract_type_encoded' in result.columns
        assert result['contract_type_encoded'].iloc[0] == 0
    
    def test_has_family_logic(self):
        df = pd.DataFrame({
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes'],
            'SeniorCitizen': [0, 0, 1]
        })
        result = engineer_features(df)
        assert 'has_family' in result.columns
        assert result['has_family'].iloc[0] == 1
        assert result['has_family'].iloc[1] == 0
        assert result['has_family'].iloc[2] == 0
    
    def test_high_risk_combo(self):
        df = pd.DataFrame({
            'InternetService': ['Fiber optic', 'DSL', 'Fiber optic'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'tenure': [10, 10, 10]
        })
        result = engineer_features(df)
        assert 'high_risk_combo' in result.columns
        assert result['high_risk_combo'].iloc[0] == 1
        assert result['high_risk_combo'].iloc[1] == 0
        assert result['high_risk_combo'].iloc[2] == 1


class TestPreprocessing:
    def test_data_split_stratified(self):
        from src.preprocessing.preprocess import split_data
        X = pd.DataFrame({'a': range(100), 'b': range(100)})
        y = pd.Series([0] * 80 + [1] * 20)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert y_train.mean() == y_test.mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])