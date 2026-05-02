# The Storyteller Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the generic Streamlit dashboard into a minimalist, narrative-driven "Storyteller" experience.

**Architecture:** Use custom CSS injection for the editorial aesthetic and a dedicated narrative builder class to generate natural language insights based on customer data and model results. Orchestrate transitions using Streamlit's state and staggered markdown rendering.

**Tech Stack:** Python, Streamlit, Plotly, Custom CSS.

---

### Task 1: CSS Foundation & Theme Overhaul

**Files:**
- Modify: `src/deployment/dashboard/app.py`

- [ ] **Step 1: Replace existing CSS with "The Storyteller" theme**

```python
# Replace the existing st.markdown style block with:
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
```

- [ ] **Step 2: Restart and verify visual baseline**
Run: `docker compose restart dashboard`
Expected: Dashboard background is pure black, fonts shift to serif style.

---

### Task 2: Narrative Builder Logic

**Files:**
- Modify: `src/deployment/dashboard/app.py`

- [ ] **Step 1: Implement NarrativeBuilder class**

```python
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
```

- [ ] **Step 2: Integrate into UI flow**
Replace the result display section (where columns col1, col2, col3 are defined) with a call to `NarrativeBuilder.build()`.

- [ ] **Step 3: Commit**
```bash
git add src/deployment/dashboard/app.py
git commit -m "feat: add narrative builder logic"
```

---

### Task 3: Interaction & Staggered Reveal

**Files:**
- Modify: `src/deployment/dashboard/app.py`

- [ ] **Step 1: Implement staggered reveal using session state**

```python
if predict_clicked:
    # ... (existing prediction logic)
    
    with st.spinner("Decoding the story..."):
        time.sleep(0.5) # Narrative tension
        narrative_html = NarrativeBuilder.build(customer_data, prediction, churn_prob, feat_names)
        st.markdown(narrative_html, unsafe_allow_html=True)
        
        # Staggered sleep for chart reveal
        time.sleep(0.5)
        st.markdown("---")
        st.markdown("### 📊 Evidence Track")
        # Render charts below...
```

- [ ] **Step 2: Final cleanup of generic Streamlit headers**
Remove `st.title("⚡ ChurnIQ")` and redundant `st.markdown("### 📊 Prediction Results")`.

- [ ] **Step 3: Verify and Commit**
Run: `docker compose restart dashboard`
Expected: Smooth fade-in of narrative text, no redundant headers.
```bash
git add src/deployment/dashboard/app.py
git commit -m "ui: implement staggered reveal and interaction polish"
```
