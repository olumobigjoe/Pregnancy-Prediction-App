import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page config MUST be the very first Streamlit call ────────────────────────
st.set_page_config(page_title="Pregnancy Outcome Predictor", layout="centered")

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #FFCOCB;
    color: #FFFFFF;
}
div.stButton > button {
    background-color: #FF00B7 !important;
    color: #FFFFFF !important;
    border-radius: 5px;
    font-weight: bold;
    width: 100%;
}
.stNumberInput input, .stSelectbox div {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load('pregnancy_outcome.pkl')

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤰 Pregnancy Outcome Prediction App")
st.divider()
st.write("Please provide the following clinical information to predict the IVF pregnancy outcome.")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input(
        "Age (years)",
        min_value=18, max_value=55, value=35,
        help="Patient age in years")

    BMI = st.number_input(
        "BMI (kg/m²)",
        min_value=10.0, max_value=60.0, value=27.0, step=0.1,
        help="Body Mass Index")

    No_Embryos = st.number_input(
        "Number of Embryos Transferred",
        min_value=0, max_value=5, value=2,
        help="How many embryos were transferred")

with col2:
    Type_Embryo = st.selectbox(
        "Type of Embryo Transferred",
        options=[("D5 — Day 5 Blastocyst", 1), ("D3 — Day 3 Cleavage", 0)],
        format_func=lambda x: x[0],
        help="D5 blastocysts generally have higher implantation rates")

    Embryo_Quality = st.selectbox(
        "Embryo Quality",
        options=[
            ("Grade 1 — Excellent", 0),
            ("Grade 2 — Good",      1),
            ("Grade 2.5 — Fair",    2),
            ("Grade 3 — Poor",      3),
            ("Grade 4 — Very Poor", 4),
        ],
        format_func=lambda x: x[0],
        help="Morphological grade assigned to the embryo")

    Sperm_Quality = st.selectbox(
        "Sperm Quality",
        options=[
            ("Grade A — Best",     0),
            ("Grade B — Good",     1),
            ("Grade C — Moderate", 2),
            ("Grade D — Poor",     3),
            ("Grade E — Very Poor",4),
        ],
        format_func=lambda x: x[0],
        help="WHO sperm quality classification")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Outcome"):

    # Build DataFrame with EXACT column names the model was trained on
    input_data = pd.DataFrame(
        [[Age,
          BMI,
          No_Embryos,
          Type_Embryo[1],       # extract encoded integer from tuple
          Embryo_Quality[1],
          Sperm_Quality[1]]],
        columns=[
            'Age',
            'BMI',
            'No_Embryos_Transferred',
            'Type_Embryo',
            'Embryo_Quality',
            'Sperm_Quality'
        ]
    )

    prediction    = model.predict(input_data)[0]
    
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ **Predicted Outcome: Pregnancy Likely**")
        
        st.balloons()
    else:
        st.error(f"❌ **Predicted Outcome: Pregnancy Unlikely**")
        

    st.divider()
    with st.expander("📋 Input summary"):
        st.dataframe(input_data)

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    "👨‍⚕️ Developed by Olumodeji Ibukun · Powered by Streamlit & Machine Learning  \n"
    "⚠️ This app is for research purposes only. Consult a gynaecologist for clinical decisions."
)
