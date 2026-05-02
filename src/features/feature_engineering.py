import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12', '12-24', '24-48', '48-72'],
            include_lowest=True
        ).astype(str)
    
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns and 'TotalCharges' in df.columns:
        df['avg_charge_per_month'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )
        df['charge_variance'] = df['MonthlyCharges'] - df['avg_charge_per_month']
    
    if 'MonthlyCharges' in df.columns and 'AvgMonthlyCharge' in df.columns:
        df['charge_increase_ratio'] = np.where(
            df['AvgMonthlyCharge'] > 0,
            df['MonthlyCharges'] / df['AvgMonthlyCharge'],
            1.0
        )
    
    if 'Contract' in df.columns:
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df['contract_type_encoded'] = df['Contract'].map(contract_map).fillna(0)
    
    if all(col in df.columns for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']):
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['num_additional_services'] = sum(
            (df[col] == 'Yes').astype(int) for col in service_cols
        )
    
    if 'InternetService' in df.columns and 'Contract' in df.columns:
        df['has_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
        df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['high_risk_combo'] = df['has_fiber'] * df['is_month_to_month']
    
    if 'tenure' in df.columns and 'Contract' in df.columns:
        df['tenure_contract_ratio'] = df['tenure'] / (
            df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two year': 24}).fillna(1)
        )
    
    if 'NumSupportCalls' in df.columns:
        df['high_support_calls'] = (df['NumSupportCalls'] > 3).astype(int)
        df['support_call_binned'] = pd.cut(
            df['NumSupportCalls'],
            bins=[-1, 1, 3, 6, 100],
            labels=['low', 'medium', 'high', 'very_high']
        ).astype(str)
    
    if 'PaymentMethod' in df.columns:
        df['is_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    if all(col in df.columns for col in ['Partner', 'Dependents']):
        df['has_family'] = (
            ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')) & 
            (df['SeniorCitizen'] == 0)
        ).astype(int)
    
    return df