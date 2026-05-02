import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

N_CUSTOMERS = 7000
CHURN_RATE = 0.26

def generate_telco_churn_data(n: int = N_CUSTOMERS, churn_rate: float = CHURN_RATE) -> pd.DataFrame:
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n + 1)]
    
    tenures = np.random.exponential(scale=24, size=n).clip(1, 72).astype(int)
    
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20]
    )
    
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
        p=[0.35, 0.20, 0.25, 0.20]
    )
    
    base_charges = np.random.normal(70, 25, n).clip(20, 120)
    monthly_charges = base_charges * (1 + np.random.uniform(-0.1, 0.15, n))
    monthly_charges = monthly_charges.round(2)
    
    total_charges = (monthly_charges * tenures).round(2)
    total_charges = np.where(total_charges == 0, np.random.uniform(50, 500, n), total_charges)
    
    genders = np.random.choice(["Male", "Female"], size=n, p=[0.5, 0.5])
    
    senior_citizen = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    
    partner = np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    dependents = np.random.choice(["Yes", "No"], size=n, p=[0.25, 0.75])
    
    phone_service = np.random.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    
    multiple_lines = np.where(
        phone_service == "Yes",
        np.random.choice(["Yes", "No"], size=n, p=[0.45, 0.55]),
        "No"
    )
    
    internet_service = np.random.choice(
        ["Fiber optic", "DSL", "No"],
        size=n,
        p=[0.45, 0.35, 0.20]
    )
    
    online_security = np.where(
        internet_service != "No",
        np.random.choice(["Yes", "No", "No internet service"], size=n, p=[0.35, 0.45, 0.20]),
        "No internet service"
    )
    online_security = np.where(internet_service == "No", "No internet service", online_security)
    
    online_backup = np.where(
        internet_service != "No",
        np.random.choice(["Yes", "No", "No internet service"], size=n, p=[0.40, 0.40, 0.20]),
        "No internet service"
    )
    online_backup = np.where(internet_service == "No", "No internet service", online_backup)
    
    device_protection = np.where(
        internet_service != "No",
        np.random.choice(["Yes", "No", "No internet service"], size=n, p=[0.40, 0.40, 0.20]),
        "No internet service"
    )
    device_protection = np.where(internet_service == "No", "No internet service", device_protection)
    
    tech_support = np.where(
        internet_service != "No",
        np.random.choice(["Yes", "No", "No internet service"], size=n, p=[0.30, 0.50, 0.20]),
        "No internet service"
    )
    tech_support = np.where(internet_service == "No", "No internet service", tech_support)
    
    streaming_tv = np.where(
        internet_service != "No",
        np.random.choice(["Yes", "No", "No internet service"], size=n, p=[0.40, 0.40, 0.20]),
        "No internet service"
    )
    streaming_tv = np.where(internet_service == "No", "No internet service", streaming_tv)
    
    streaming_movies = np.where(
        internet_service != "No",
        np.random.choice(["Yes", "No", "No internet service"], size=n, p=[0.42, 0.38, 0.20]),
        "No internet service"
    )
    streaming_movies = np.where(internet_service == "No", "No internet service", streaming_movies)
    
    num_support_calls = np.random.poisson(lam=1.5, size=n).clip(0, 15)
    
    avg_monthly_charges = np.random.normal(65, 20, n).clip(20, 115).round(2)
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenures,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract_types,
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n, p=[0.60, 0.40]),
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "NumSupportCalls": num_support_calls,
        "AvgMonthlyCharge": avg_monthly_charges
    })
    
    churn_prob = (
        0.20 * (contract_types == "Month-to-month").astype(float)
        + 0.18 * (tenures < 12).astype(float)
        + 0.08 * (internet_service == "Fiber optic").astype(float)
        + 0.06 * (payment_methods == "Electronic check").astype(float)
        + 0.05 * (num_support_calls > 3).astype(float)
        + np.random.uniform(-0.08, 0.08, n)
    )
    churn_prob = churn_prob.clip(0.05, 0.5)
    
    df["Churn"] = np.where(np.random.random(n) < churn_prob, "Yes", "No")
    
    return df


if __name__ == "__main__":
    df = generate_telco_churn_data()
    
    output_path = Path("/home/ayyan/project/intership/data/raw/synthetic_telco_churn.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} customer records")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True).round(3)}")
    print(f"\nSample:\n{df.head().to_string()}")