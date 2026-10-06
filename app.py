import os
import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DiaCheck · Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finalModel.pkl")

# Exact feature order used when the model was trained (pd.get_dummies, drop_first=True)
FEATURE_ORDER = [
    "age", "hypertension", "heart_disease", "bmi", "HbA1c_level",
    "blood_glucose_level", "gender_Male", "gender_Other",
    "smoking_history_current", "smoking_history_ever", "smoking_history_former",
    "smoking_history_never", "smoking_history_not current",
]

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #e0f2fe 0%, transparent 60%),
                radial-gradient(1000px 500px at 100% 0%, #ede9fe 0%, transparent 55%),
                #f8fafc;
}
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 55%, #8b5cf6 100%);
    border-radius: 24px; padding: 2.4rem 2.6rem; color: #fff;
    box-shadow: 0 20px 45px -15px rgba(99,102,241,.55);
    position: relative; overflow: hidden; margin-bottom: 1.6rem;
}
.hero::after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
    border-radius: 50%; background: rgba(255,255,255,.12);
}
.hero::before {
    content: ""; position: absolute; right: 120px; bottom: -90px; width: 200px; height: 200px;
    border-radius: 50%; background: rgba(255,255,255,.08);
}
.hero h1 { font-size: 2.3rem; font-weight: 800; margin: 0 0 .4rem 0; letter-spacing: -.5px; color:#fff; }
.hero p { font-size: 1.02rem; opacity: .92; margin: 0; max-width: 640px; line-height: 1.55; }
.badge {
    display: inline-block; background: rgba(255,255,255,.2); padding: .3rem .8rem;
    border-radius: 999px; font-size: .78rem; font-weight: 600; margin-bottom: .9rem;
    backdrop-filter: blur(6px);
}

/* Cards */
.card {
    background: #fff; border-radius: 20px; padding: 1.6rem 1.7rem;
    border: 1px solid #e2e8f0; box-shadow: 0 10px 30px -18px rgba(15,23,42,.25);
    margin-bottom: 1.2rem;
}
.card h3 { margin: 0 0 .2rem 0; font-size: 1.15rem; font-weight: 700; color: #0f172a; }
.card .sub { color: #64748b; font-size: .88rem; margin-bottom: 1rem; }

/* Metric tiles */
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin-top: .6rem; }
.tile {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: .9rem 1rem; text-align: center;
}
.tile .v { font-size: 1.35rem; font-weight: 800; color: #0f172a; }
.tile .l { font-size: .74rem; color: #64748b; text-transform: uppercase; letter-spacing: .6px; font-weight: 600; }

/* Result */
.result { border-radius: 20px; padding: 1.8rem; text-align: center; border: 1px solid; }
.result.low  { background: linear-gradient(160deg,#ecfdf5,#d1fae5); border-color:#6ee7b7; }
.result.high { background: linear-gradient(160deg,#fef2f2,#fee2e2); border-color:#fca5a5; }
.result .emoji { font-size: 2.6rem; }
.result .title { font-size: 1.5rem; font-weight: 800; margin: .3rem 0 .2rem; }
.result.low .title  { color:#047857; }
.result.high .title { color:#b91c1c; }
.result .desc { color:#475569; font-size:.92rem; }
.prob { font-size: 3.2rem; font-weight: 800; line-height: 1; margin: .8rem 0 .2rem; color:#0f172a; }
.prob-label { font-size: .78rem; text-transform: uppercase; letter-spacing: 1px; color:#64748b; font-weight:600; }

.bar-wrap { background:#e2e8f0; border-radius: 999px; height: 14px; margin: 1rem 0 .4rem; overflow:hidden; }
.bar { height:100%; border-radius:999px; }
.bar-scale { display:flex; justify-content:space-between; font-size:.72rem; color:#94a3b8; font-weight:600; }

/* Factor rows */
.factor { display:flex; align-items:center; justify-content:space-between; padding:.65rem .9rem;
    border-radius:12px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:.5rem; font-size:.9rem; }
.factor .n { font-weight:600; color:#0f172a; }
.factor .s { font-size:.75rem; font-weight:700; padding:.2rem .6rem; border-radius:999px; }
.s.ok   { background:#d1fae5; color:#047857; }
.s.warn { background:#fef3c7; color:#b45309; }
.s.bad  { background:#fee2e2; color:#b91c1c; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #fff !important; }
section[data-testid="stSidebar"] hr { border-color: #1e293b; }

/* Button */
div.stButton > button {
    width: 100%; border: 0; border-radius: 14px; padding: .85rem 1rem; font-weight: 700; font-size: 1rem;
    color: #fff; background: linear-gradient(135deg,#0ea5e9,#6366f1);
    box-shadow: 0 12px 24px -10px rgba(99,102,241,.7); transition: all .2s ease;
}
div.stButton > button:hover { transform: translateY(-2px); filter: brightness(1.07); color:#fff; border:0; }
div.stButton > button:active { transform: translateY(0); }

.disclaimer { font-size:.8rem; color:#64748b; text-align:center; margin-top:1.4rem; line-height:1.5; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_features(age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking):
    row = {c: 0 for c in FEATURE_ORDER}
    row.update(
        age=age,
        hypertension=int(hypertension),
        heart_disease=int(heart_disease),
        bmi=bmi,
        HbA1c_level=hba1c,
        blood_glucose_level=glucose,
    )
    if gender == "Male":
        row["gender_Male"] = 1
    elif gender == "Other":
        row["gender_Other"] = 1
    # "No Info" is the dropped baseline category
    if smoking != "No Info":
        row[f"smoking_history_{smoking}"] = 1
    return pd.DataFrame([row])[FEATURE_ORDER]


def bmi_status(b):
    if b < 18.5: return "Underweight", "warn"
    if b < 25:   return "Normal", "ok"
    if b < 30:   return "Overweight", "warn"
    return "Obese", "bad"


def hba1c_status(h):
    if h < 5.7: return "Normal", "ok"
    if h < 6.5: return "Pre-diabetic range", "warn"
    return "Diabetic range", "bad"


def glucose_status(g):
    if g < 140: return "Normal", "ok"
    if g < 200: return "Elevated", "warn"
    return "High", "bad"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 DiaCheck")
    st.caption("AI-assisted diabetes risk screening")
    st.markdown("---")
    st.markdown("### 🧬 Patient Profile")

    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.slider("Age (years)", 1, 100, 45)
    smoking = st.selectbox(
        "Smoking history",
        ["No Info", "never", "former", "current", "not current", "ever"],
        format_func=lambda x: x.capitalize() if x != "No Info" else x,
    )

    st.markdown("### 🫀 Medical History")
    hypertension = st.toggle("Hypertension")
    heart_disease = st.toggle("Heart disease")

    st.markdown("### 🔬 Clinical Measurements")
    bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 27.0, 0.1)
    hba1c = st.slider("HbA1c level (%)", 3.5, 9.0, 5.8, 0.1)
    glucose = st.slider("Blood glucose (mg/dL)", 70, 300, 120, 1)

    st.markdown("---")
    st.caption("Model · Logistic Regression\n\nDataset · 100,000 patient records")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <div class="badge">⚡ Machine Learning · Logistic Regression</div>
    <h1>Diabetes Risk Predictor</h1>
    <p>Enter a patient's health information in the sidebar and get an instant,
    data-driven estimate of diabetes risk — along with a breakdown of key health indicators.</p>
</div>
""",
    unsafe_allow_html=True,
)

model = load_model()
if model is None:
    st.error(
        "**`finalModel.pkl` not found.** Run your notebook's save cell "
        "(`joblib.dump(clf, 'finalModel.pkl')`) and place the file in the same folder as `app.py`."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
left, right = st.columns([1.05, 1], gap="large")

with left:
    b_lbl, b_cls = bmi_status(bmi)
    h_lbl, h_cls = hba1c_status(hba1c)
    g_lbl, g_cls = glucose_status(glucose)

    st.markdown(
        f"""
<div class="card">
    <h3>📋 Patient Summary</h3>
    <div class="sub">Live overview of the values you've entered</div>
    <div class="tiles">
        <div class="tile"><div class="v">{age}</div><div class="l">Age</div></div>
        <div class="tile"><div class="v">{bmi:.1f}</div><div class="l">BMI</div></div>
        <div class="tile"><div class="v">{hba1c:.1f}%</div><div class="l">HbA1c</div></div>
        <div class="tile"><div class="v">{glucose}</div><div class="l">Glucose</div></div>
        <div class="tile"><div class="v">{'Yes' if hypertension else 'No'}</div><div class="l">Hypertension</div></div>
        <div class="tile"><div class="v">{'Yes' if heart_disease else 'No'}</div><div class="l">Heart disease</div></div>
    </div>
</div>

<div class="card">
    <h3>🔍 Health Indicators</h3>
    <div class="sub">Standard clinical reference ranges</div>
    <div class="factor"><span class="n">Body Mass Index · {bmi:.1f}</span><span class="s {b_cls}">{b_lbl}</span></div>
    <div class="factor"><span class="n">HbA1c · {hba1c:.1f}%</span><span class="s {h_cls}">{h_lbl}</span></div>
    <div class="factor"><span class="n">Blood Glucose · {glucose} mg/dL</span><span class="s {g_cls}">{g_lbl}</span></div>
    <div class="factor"><span class="n">Hypertension</span><span class="s {'bad' if hypertension else 'ok'}">{'Present' if hypertension else 'Absent'}</span></div>
    <div class="factor"><span class="n">Heart Disease</span><span class="s {'bad' if heart_disease else 'ok'}">{'Present' if heart_disease else 'Absent'}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="card"><h3>🎯 Prediction</h3>'
        '<div class="sub">Click the button to analyse the current profile</div></div>',
        unsafe_allow_html=True,
    )
    run = st.button("🔮  Analyse Risk")

    if run:
        X = build_features(age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking)
        with st.spinner("Analysing patient data..."):
            proba = float(model.predict_proba(X)[0][1])
            pred = int(model.predict(X)[0])

        pct = proba * 100
        if pred == 1:
            cls, emoji, title = "high", "⚠️", "High Risk of Diabetes"
            desc = "The model indicates a likelihood of diabetes. Please consult a healthcare professional for confirmation."
        else:
            cls, emoji, title = "low", "✅", "Low Risk of Diabetes"
            desc = "The model does not indicate diabetes for this profile. Maintain a healthy lifestyle and regular check-ups."

        if pct < 30:   color = "linear-gradient(90deg,#34d399,#10b981)"
        elif pct < 60: color = "linear-gradient(90deg,#fbbf24,#f59e0b)"
        else:          color = "linear-gradient(90deg,#f87171,#dc2626)"

        st.markdown(
            f"""
<div class="result {cls}">
    <div class="emoji">{emoji}</div>
    <div class="title">{title}</div>
    <div class="prob">{pct:.1f}%</div>
    <div class="prob-label">Predicted probability</div>
    <div class="bar-wrap"><div class="bar" style="width:{max(pct,2):.1f}%; background:{color};"></div></div>
    <div class="bar-scale"><span>0%</span><span>50%</span><span>100%</span></div>
    <p class="desc" style="margin-top:1rem;">{desc}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("View model input"):
            st.dataframe(X, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
<div class="card" style="text-align:center; padding:2.6rem 1.5rem;">
    <div style="font-size:3rem;">🧪</div>
    <div style="font-weight:700; color:#0f172a; margin-top:.4rem;">Awaiting analysis</div>
    <div style="color:#64748b; font-size:.9rem; margin-top:.3rem;">
        Fill in the patient profile and press <b>Analyse Risk</b>.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    """
<div class="disclaimer">
    ⚕️ <b>Disclaimer:</b> This tool is for educational and screening purposes only and is not a substitute
    for professional medical advice, diagnosis or treatment.
</div>
""",
    unsafe_allow_html=True,
)