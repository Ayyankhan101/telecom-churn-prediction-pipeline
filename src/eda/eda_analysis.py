import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
OUTPUT_DIR = Path("/home/ayyan/project/intership/reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "/home/ayyan/project/intership/data/raw/synthetic_telco_churn.csv"

def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df

def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
           colors=colors, explode=(0, 0.05), startangle=90)
    ax.set_title('Churn Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'churn_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: churn_distribution.png")

def plot_churn_by_contract(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).sort_values(ascending=False)
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(contract_churn.index, contract_churn.values, color=colors)
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_xlabel('Contract Type', fontsize=12)
    ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, contract_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'churn_by_contract.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: churn_by_contract.png")

def plot_churn_by_tenure(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                                 labels=['0-12', '12-24', '24-48', '48-72'])
    tenure_churn = df.groupby('tenure_group')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    )
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tenure_churn)))
    bars = ax.bar(tenure_churn.index.astype(str), tenure_churn.values, color=colors)
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_xlabel('Tenure (months)', fontsize=12)
    ax.set_title('Churn Rate by Tenure', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, tenure_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'churn_by_tenure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: churn_by_tenure.png")

def plot_churn_by_payment(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    payment_churn = df.groupby('PaymentMethod')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).sort_values(ascending=False)
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax.barh(payment_churn.index, payment_churn.values, color=colors)
    ax.set_xlabel('Churn Rate (%)', fontsize=12)
    ax.set_title('Churn Rate by Payment Method', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, payment_churn.values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'churn_by_payment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: churn_by_payment.png")

def plot_correlation_heatmap(df):
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumSupportCalls']
    df_corr = df[numerical_cols].copy()
    df_corr['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df_corr.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap.png")

def plot_churn_by_services(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    services = ['InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod']
    titles = ['Internet Service', 'Online Security', 'Tech Support', 'Payment Method']
    
    for ax, service, title in zip(axes.flatten(), services, titles):
        service_churn = df.groupby(service)['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(service_churn)))
        ax.bar(range(len(service_churn)), service_churn.values, color=colors)
        ax.set_xticks(range(len(service_churn)))
        ax.set_xticklabels(service_churn.index, rotation=45, ha='right')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title(f'Churn by {title}')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'churn_by_services.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: churn_by_services.png")

def plot_monthly_charges_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    for churn_val, color, label in [('No', '#2ecc71', 'No Churn'), ('Yes', '#e74c3c', 'Churn')]:
        subset = df[df['Churn'] == churn_val]['MonthlyCharges']
        ax1.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    ax1.set_xlabel('Monthly Charges ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Monthly Charges by Churn Status')
    ax1.legend()
    
    ax2 = axes[1]
    for churn_val, color, label in [('No', '#2ecc71', 'No Churn'), ('Yes', '#e74c3c', 'Churn')]:
        subset = df[df['Churn'] == churn_val]['tenure']
        ax2.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    ax2.set_xlabel('Tenure (months)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Tenure by Churn Status')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'charges_tenure_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: charges_tenure_dist.png")

def generate_eda_report(df):
    print("\n" + "="*50)
    print("EDA SUMMARY")
    print("="*50)
    
    print(f"\nDataset: {len(df)} customers, {df.shape[1]} features")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean()*100:.1f}%")
    
    print("\nTop Churn Risk Factors:")
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
    print(f"  - Month-to-month: {contract_churn.get('Month-to-month', 0)*100:.1f}% churn")
    print(f"  - Tenure < 12 months: {df[df['tenure'] < 12]['Churn'].apply(lambda x: x=='Yes').mean()*100:.1f}% churn")
    print(f"  - Electronic check: {df[df['PaymentMethod'] == 'Electronic check']['Churn'].apply(lambda x: x=='Yes').mean()*100:.1f}% churn")
    print(f"  - Fiber optic: {df[df['InternetService'] == 'Fiber optic']['Churn'].apply(lambda x: x=='Yes').mean()*100:.1f}% churn")
    
    print("\nLow Churn Factors:")
    print(f"  - Two year contract: {contract_churn.get('Two year', 0)*100:.1f}% churn")
    print(f"  - Tenure > 48 months: {df[df['tenure'] > 48]['Churn'].apply(lambda x: x=='Yes').mean()*100:.1f}% churn")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    df = load_raw_data()
    
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure(df)
    plot_churn_by_payment(df)
    plot_correlation_heatmap(df)
    plot_churn_by_services(df)
    plot_monthly_charges_distribution(df)
    
    generate_eda_report(df)