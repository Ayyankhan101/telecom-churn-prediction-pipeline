# Design Spec: ChurnIQ "The Storyteller" Dashboard

**Date**: 2026-05-02
**Author**: Gemini CLI (Senior Software Engineer)
**Status**: Approved

## 1. Objective
Transform the existing generic Streamlit dashboard into a high-impact, narrative-driven experience called "The Storyteller". Focus on customer journey and natural language insights over raw data tables.

## 2. Aesthetic Direction: "The Storyteller"
- **Tone**: Minimalist, editorial, serious.
- **Color Palette**: Deep Charcoal (#0a0a0a), muted Grey (#888), pure White (#fff) for highlights. 
- **Accents**: Risk Red (#ff4b2b) and Retention Blue (#00d2ff).
- **Typography**: Serif fonts (Georgia, Playfair Display) for headlines to evoke a "narrative" or "report" feel. Sans-serif for data points.

## 3. Architecture & Components

### 3.1. Layout
- **Sidebar**: Clean, grouped input forms (Identity, Services, Billing). Use a "Less is More" approach with labels.
- **Main Stage**:
    - **Header**: Large, minimalist title "ChurnIQ".
    - **The Verdict**: A central focal point. Massive, uppercase text for the prediction (e.g., "LIKELY CHURN").
    - **Dynamic Narrative**: A paragraph of text that changes based on input data and feature importance. Use placeholders like `[risk_factor]` to highlight specific customer attributes.
    - **Evidence Track**: A visual timeline or set of "chapters" showing the key pillars of the customer's profile (Tenure, Contract, Services).

### 3.2. Dynamic Narrative Logic
The dashboard will construct a sentence based on the top 3 feature importance values.
Example: 
> "This customer is **at risk**. Their story is defined by a **Month-to-month contract** and **Fiber optic service**. Despite being with us for **[tenure] months**, they have reached out for support **[num] times**."

## 4. Interaction Design
- **Mood Pulse**: The background of the main card will subtly pulse with the accent color (Red for Churn, Blue for Stay) after prediction.
- **Staggered Reveal**: Components will fade in sequentially:
    1. The Verdict (0s)
    2. The Narrative (0.5s)
    3. Visual Evidence (1.0s)
- **High-Impact Transitions**: Smooth transitions between the "Empty" state and the "Result" state to maintain the user's focus.

## 5. Technical Implementation
- **Frontend**: Streamlit + Custom CSS (`st.markdown` with `unsafe_allow_html=True`).
- **Logic**: Python-based narrative builder in `src/deployment/dashboard/app.py`.
- **API Integration**: Maintain existing connection to Flask API on port 5000.
- **Data Flow**: Sidebar inputs -> API Payload -> Prediction + Feature Importance -> Narrative Builder -> UI Display.

## 6. Testing & Validation
- **Visual Check**: Verify CSS renders correctly in Chrome/Firefox.
- **Narrative Check**: Ensure the generated text makes grammatical sense for various customer profiles.
- **Performance**: Ensure animations don't lag the Streamlit re-run cycle.
