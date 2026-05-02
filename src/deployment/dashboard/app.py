import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.deployment.api.shared import preprocess_input
from src.deployment.api.utils import FEATURE_ORDER

# ============================================================================
# Model Loading (Self-Contained)
# ============================================================================
@st.cache_resource
def load_models():
    """Load and cache the models for performance."""
    model_dir = Path(__file__).resolve().parents[3] / "models"
    try:
        model = joblib.load(model_dir / "best_model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        encoders = joblib.load(model_dir / "encoders.joblib")
        
        # Load version info
        version_file = model_dir / "version.json"
        version_data = {}
        if version_file.exists():
            import json
            with open(version_file) as f:
                version_data = json.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'encoders': encoders,
            'version': version_data.get('version', '1.0.0'),
            'status': 'loaded'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# ============================================================================
# Local Prediction Helper
# ============================================================================
def predict_local(customer_data, models_dict):
    """Perform prediction using loaded models."""
    if models_dict['status'] != 'loaded':
        return {'error': models_dict.get('error', 'Models not loaded')}
    
    try:
        model = models_dict['model']
        scaler = models_dict['scaler']
        encoders = models_dict['encoders']
        
        # Preprocess
        X = preprocess_input(customer_data, encoders=encoders, scaler=scaler)
        
        # Predict
        prediction_val = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Feature Importance (Local)
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = np.zeros(len(FEATURE_ORDER))

        feat_imp = dict(zip(FEATURE_ORDER, importance))
        sorted_features = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        return {
            'prediction': 'Churn' if prediction_val == 1 else 'No Churn',
            'churn_probability': float(probability[1]),
            'no_churn_probability': float(probability[0]),
            'top_features': [f[0] for f in sorted_features],
            'importance_values': [float(f[1]) for f in sorted_features]
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# Custom Theme Configuration
# ============================================================================
st.set_page_config(
    page_title="ChurnIQ - Telecom Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .narrative-title {
        font-family: 'Playfair Display', serif;
        color: #ffffff !important;
    }

    .main-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 40px;
        border-radius: 4px;
        margin-top: 20px;
        transition: all 0.5s ease;
    }

    .verdict-text {
        font-size: 4rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.2rem;
        margin-bottom: 10px;
        animation: fadeIn 1s ease-in;
    }

    .risk-high { color: #ff4b2b; text-shadow: 0 0 20px rgba(255, 75, 43, 0.3); }
    .risk-low { color: #00d2ff; text-shadow: 0 0 20px rgba(0, 210, 255, 0.3); }

    .narrative-body {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        line-height: 1.8;
        color: #bbb;
        max-width: 800px;
        margin: 30px 0;
        animation: fadeIn 1.5s ease-in;
    }

    .highlight {
        color: #fff;
        font-weight: 600;
        border-bottom: 1px solid #444;
        padding-bottom: 2px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e0e0e;
        border-right: 1px solid #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None


# ============================================================================
# Chart Functions
# ============================================================================
def create_gauge_chart(churn_prob, no_churn_prob):
    """Create radial gauge chart for churn probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 20, 'color': '#e8e8e8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#e8e8e8'},
            'bar': {'color': "#ff6b6b" if churn_prob > 0.5 else "#00d4aa"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 212, 170, 0.2)'},
                {'range': [50, 100], 'color': 'rgba(255, 107, 107, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        number={'font': {'size': 36, 'color': '#e8e8e8'}, 'suffix': "%"}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e8e8e8'}
    )
    
    return fig


def create_feature_importance_chart(feature_names, importance_values):
    """Create horizontal bar chart for top 5 feature importance"""
    fig = go.Figure()
    
    # Color bars based on impact direction (positive = churn risk)
    colors = ['#ff6b6b' if imp > 0 else '#00d4aa' for imp in importance_values]
    
    fig.add_trace(go.Bar(
        y=feature_names,
        x=importance_values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{abs(val):.3f}" for val in importance_values],
        textposition='auto',
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Top 5 Churn Risk Factors',
            'font': {'size': 18, 'color': '#e8e8e8'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Impact on Churn',
        yaxis_title='Feature',
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e8e8e8'},
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            categoryorder='total ascending'
        )
    )
    
    return fig


# ============================================================================
# Narrative Builder
# ============================================================================
class NarrativeBuilder:
    @staticmethod
    def build(data, prediction, prob, top_features):
        risk_level = "at risk" if prediction == "Churn" else "stable"
        verdict_class = "risk-high" if prediction == "Churn" else "risk-low"
        
        # Build story components
        contract = data.get('Contract', 'Unknown')
        tenure = data.get('tenure', 0)
        support = data.get('NumSupportCalls', 0)
        internet = data.get('InternetService', 'No')
        
        story = f"""
        <div class='main-card'>
            <div class='narrative-title'>Will they stay?</div>
            <div class='verdict-text {verdict_class}'>{prediction.upper()}</div>
            <div class='narrative-body'>
                This customer is <span class='highlight'>{risk_level}</span>. 
                Their story is defined by a <span class='highlight'>{contract} contract</span> 
                and <span class='highlight'>{internet} internet service</span>. 
                Despite being with us for <span class='highlight'>{tenure} months</span>, 
                they have already reached out for support <span class='highlight'>{support} times</span>.
                <br><br>
                The primary factors driving this outcome are <span class='highlight'>{", ".join(top_features[:3])}</span>.
            </div>
        </div>
        """
        return story


# ============================================================================
# Sidebar - Local Status
# ============================================================================
models_dict = load_models()

with st.sidebar:
    st.markdown("### ⚡ System Status")
    
    if models_dict['status'] == 'loaded':
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00d4aa22, #00b89422);
            border: 1px solid #00d4aa;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        ">
            <span style="font-size: 2rem;">🟢</span><br>
            <strong style="color: #00d4aa;">System Ready</strong><br>
            <small style="color: #888;">Models Loaded Locally</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 8px;">
            <small style="color: #888;">Model Version</small><br>
            <strong style="color: #e8e8e8;">{models_dict.get('version', 'N/A')}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ff6b6b22, #ff8e5322);
            border: 1px solid #ff6b6b;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        ">
            <span style="font-size: 2rem;">🔴</span><br>
            <strong style="color: #ff6b6b;">Error</strong><br>
            <small style="color: #888;">Failed to load models</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.error(f"Load Error: {models_dict.get('error')}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <small style="color: #888;">
    <strong>ChurnIQ</strong> uses machine learning to predict customer churn risk.<br><br>
    Enter customer details to get instant predictions with probability scores.
    </small>
    """, unsafe_allow_html=True)


# ============================================================================
# Main Content
# ============================================================================
# Generic headers removed for "The Storyteller" editorial layout

# Customer input form
st.markdown("#### 📝 Customer Details")
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"], help="Customer has a partner")
        dependents = st.selectbox("Has Dependents", ["Yes", "No"], help="Customer has dependents")
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="Months with the company")
    
    with c2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        if phone == "Yes":
            multiline = st.selectbox("Multiple Lines", ["Yes", "No"])
        else:
            multiline = "No phone service"
            
        internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"], help="Internet service type")
        
        if internet != "No":
            security = st.selectbox("Online Security", ["Yes", "No"])
            backup = st.selectbox("Online Backup", ["Yes", "No"])
            protection = st.selectbox("Device Protection", ["Yes", "No"])
            support = st.selectbox("Tech Support", ["Yes", "No"])
            tv = st.selectbox("Streaming TV", ["Yes", "No"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        else:
            security = backup = protection = support = tv = movies = "No internet service"
    
    st.markdown("#### 💳 Billing Information")
    b1, b2, b3 = st.columns(3)
    with b1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="Contract type")
    with b2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with b3:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    m1, m2, m3 = st.columns(3)
    with m1:
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
    with m2:
        total = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)
    with m3:
        support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2)
    
    avg_charge = monthly if tenure == 0 else total / tenure if total > 0 else monthly

# Prediction button
predict_clicked = st.button("⚡ Predict Churn", type="primary", use_container_width=True)

if predict_clicked:
    if models_dict['status'] == 'error':
        st.error("Cannot perform prediction: Models not loaded.")
    else:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiline,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "NumSupportCalls": support_calls,
            "AvgMonthlyCharge": avg_charge
        }
        
        with st.spinner("Decoding the story..."):
            time.sleep(1.2) # Anticipation
            result = predict_local(customer_data, models_dict)
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            prediction = result.get('prediction', 'Unknown')
            churn_prob = result.get('churn_probability', 0)
            no_churn_prob = result.get('no_churn_probability', 0)
            feat_names = result.get('top_features', [])
            feat_vals = result.get('importance_values', [])
            
            st.session_state.feature_importance = (feat_names, feat_vals)

            # Reveal 1: Narrative
            narrative_html = NarrativeBuilder.build(customer_data, prediction, churn_prob, feat_names)
            st.markdown(narrative_html, unsafe_allow_html=True)
            
            # Reveal 2: Evidence Track
            time.sleep(0.8)
            st.markdown("---")
            st.markdown("### 🔍 Evidence Track")
            
            # Store prediction in history
            st.session_state.prediction_history.append({
                'prediction': prediction,
                'churn_prob': churn_prob,
                'no_churn_prob': no_churn_prob,
                'timestamp': pd.Timestamp.now()
            })
            
            # Feature Importance Section
            st.markdown("---")
            st.markdown("### 🔍 Risk Factor Analysis")
            
            # Additional Analytical Charts
            st.markdown("### 📊 Billing & Loyalty Analysis")
            c_ext1, c_ext2 = st.columns(2)
            
            with c_ext1:
                # Loyalty vs Charges Plot
                fig_loyalty = px.scatter(
                    x=[tenure], y=[monthly],
                    labels={"x": "Tenure (Months)", "y": "Monthly Charges ($)"},
                    title="Current Customer: Charges vs Loyalty",
                    range_x=[0, 72], range_y=[0, 150],
                    color_discrete_sequence=["#00d4aa" if prediction != "Churn" else "#ff6b6b"]
                )
                fig_loyalty.update_traces(marker=dict(size=20, symbol="star"))
                fig_loyalty.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#e8e8e8'})
                st.plotly_chart(fig_loyalty, use_container_width=True)

            with c_ext2:
                # Service Usage Radar-style Bar
                services = ['Security', 'Backup', 'Protection', 'Support', 'TV', 'Movies']
                usage = [1 if val == "Yes" else 0 for val in [security, backup, protection, support, tv, movies]]
                fig_services = px.bar(
                    x=services, y=usage,
                    title="Value-Added Services Adopted",
                    color=services,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_services.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#e8e8e8'}, yaxis_range=[0,1])
                st.plotly_chart(fig_services, use_container_width=True)
            
            if st.session_state.feature_importance:
                col_bar1, col_bar2 = st.columns([3, 1])
                with col_bar1:
                    feat_names, feat_vals = st.session_state.feature_importance
                    bar_fig = create_feature_importance_chart(feat_names, feat_vals)
                    st.plotly_chart(bar_fig, use_container_width=True)
                with col_bar2:
                    st.markdown("""
                    <div style="padding: 20px; background: rgba(255,255,255,0.03); border-radius: 12px; height: 350px; display: flex; flex-direction: column; justify-content: center;">
                        <h4 style="color: #e8e8e8; margin-bottom: 15px;">Understanding Impact</h4>
                        <p style="color: #888; line-height: 1.6;">
                            Positive values (coral bars) increase churn risk.
                            Negative values (teal bars) indicate protective factors that reduce churn.
                        </p>
                        <p style="color: #888; line-height: 1.6; margin-top: 15px;">
                            Focus on the top factors for maximum retention impact.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# Prediction History (Collapsible)
# ============================================================================
if st.session_state.prediction_history:
    st.markdown("---")
    with st.expander("📈 Prediction History", expanded=False):
        if len(st.session_state.prediction_history) > 0:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.markdown(f"**Total predictions:** {len(history_df)}")
            
            # Summary stats
            churn_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 'Churn')
            st.markdown(f"**Churn rate:** {churn_count/len(history_df)*100:.1f}%")
            
            # Simple line chart of churn probability over time
            if len(history_df) > 1:
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['churn_prob'],
                    mode='lines+markers',
                    name='Churn Probability',
                    line=dict(color='#ff6b6b', width=3),
                    marker=dict(size=8, color='#ff6b6b'),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.2)'
                ))
                fig_history.update_layout(
                    title="Churn Probability Trend",
                    xaxis_title="Prediction #",
                    yaxis_title="Churn Probability",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e8e8e8'},
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0,1])
                )
                st.plotly_chart(fig_history, use_container_width=True)

# ============================================================================
# Risk Factors Section (Kept as reference)
# ============================================================================
st.markdown("---")
st.markdown("### 🔍 Key Churn Indicators")

st.markdown("""
| Factor | Risk Level | Impact |
|--------|------------|--------|
| Month-to-month contract | High | 2x higher churn than multi-year |
| New customer (<12 months) | High | Highest churn period |
| Fiber optic internet | Medium | Premium but higher complaints |
| Electronic check payment | Medium | 28% churn rate |
| 3+ support calls | High | Strong churn signal |
""")

# Tips
st.info("💡 **Retention Tips:** Focus on new customers in first 12 months. Offer contract upgrades. Promote auto-pay to reduce churn risk.")
