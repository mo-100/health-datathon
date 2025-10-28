import streamlit as st
import json, os, dotenv
from openai import Client
import pandas as pd
from core.extractors import extract_ctg_from_pdf, extract_miscarriage_from_pdf
from core.embeddings import load_embedding_model, precompute_doc_embeddings, query_docs
from core.predictors import load_ctg_model, predict_miscarriage, predict_ctg, run_risk_system_ctg, run_risk_system_miscarriage, load_miscarriage_model
from core.widgets import render_report_dashboard

# ------------------ SETUP ------------------ #
dotenv.load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = Client(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

st.set_page_config(page_title="PreSafe", layout="wide")

# ------------------ LOAD MODELS ------------------ #
ctg_model = st.cache_resource(load_ctg_model)()
miscarriage_model = st.cache_resource(load_miscarriage_model)()
with open('ml/data/advices.jsonl') as f:
    advice_docs = json.loads(f.read())
tokenizer, emb_model = st.cache_resource(load_embedding_model)()
doc_embeddings = st.cache_resource(precompute_doc_embeddings)(advice_docs, emb_model, tokenizer)

# ------------------ PAGE NAVIGATION ------------------ #
if "page" not in st.session_state:
    st.session_state.page = "intro"

# ------------------ PAGE 1: PATIENT INFO ------------------ #
if st.session_state.page == "intro":
    st.title("👩‍⚕️ Patient Information")

    st.markdown("Please enter the patient details below to continue:")

    patient_id = st.text_input("Patient ID")
    gestational_age = st.number_input("Gestational Age (weeks)", 4.0, 42.0, step=0.1)

    if st.button("Next"):
        if not patient_id or gestational_age <= 0:
            st.warning("Please fill in both fields.")
        else:
            st.session_state.patient_id = patient_id
            st.session_state.gestational_age = gestational_age
            # Navigate to appropriate page
            st.session_state.page = "ctg" if gestational_age >= 26 else "miscarriage"
            st.rerun()

# ------------------ PAGE 2A: CTG TEST ------------------ #
# ------------------ PAGE 2A: CTG TEST ------------------ #
elif st.session_state.page == "ctg":
    st.title("🫀 CTG Risk Assessment")
    st.caption(f"Patient ID: {st.session_state.patient_id} | GA: {st.session_state.gestational_age} weeks")

    with st.sidebar:
        st.header("📎 Upload CTG Report (PDF)")
        ctg_pdf = st.file_uploader("Upload CTG Report (PDF)", type=["pdf"])

        if ctg_pdf:
            try:
                ctg_features = extract_ctg_from_pdf(ctg_pdf, client)
                st.session_state.ctg_features = ctg_features  # ✅ store extracted data
                st.success("✅ CTG features extracted successfully and fields auto-filled.")
            except Exception as e:
                st.error(f"CTG extraction failed: {e}")

    st.header("CTG Input (Manual Entry or Auto-filled from PDF)")

    ctg_columns = [
        "baseline_value", "accelerations", "fetal_movement", "uterine_contractions",
        "light_decelerations", "severe_decelerations", "prolongued_decelerations",
        "abnormal_short_term_variability", "mean_value_of_short_term_variability",
        "percentage_of_time_with_abnormal_long_term_variability",
        "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
        "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes",
        "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance",
        "histogram_tendency"
    ]

    col1, col2, col3 = st.columns(3)
    ctg_inputs = {}

    # Use extracted values if available
    prefilled = st.session_state.get("ctg_features", {})

    for i, feature in enumerate(ctg_columns):
        col = [col1, col2, col3][i % 3]
        with col:
            default_val = prefilled.get(feature, 0.0)
            ctg_inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(default_val) if isinstance(default_val, (int, float)) else 0.0,
                step=0.1,
                format="%.3f"
            )

    if st.button("Run CTG Assessment"):
        ctg_df = pd.DataFrame([ctg_inputs])
        ctg_output = predict_ctg(ctg_model, ctg_df)
        query = f"CTG Prediction: {ctg_output['predicted_class']}, top features: {ctg_output['top_features']}, recommendations: {ctg_output['recommendations']}"
        top_advices = query_docs(query, doc_embeddings, emb_model, tokenizer, advice_docs)
        report = run_risk_system_ctg(top_advices, ctg_output, client)
        render_report_dashboard(report, test_type="CTG")

    if st.button("⬅ Back"):
        st.session_state.page = "intro"
        st.rerun()

# ------------------ PAGE 2B: MISCARRIAGE TEST ------------------ #
# ------------------ PAGE 2B: MISCARRIAGE TEST ------------------ #
elif st.session_state.page == "miscarriage":
    st.title("🍼 Miscarriage Risk Assessment")
    st.caption(f"Patient ID: {st.session_state.patient_id} | GA: {st.session_state.gestational_age} weeks")

    with st.sidebar:
        st.header("📎 Upload Miscarriage Ultrasound Report (PDF)")
        miscarriage_pdf = st.file_uploader("Upload Ultrasound Report (PDF)", type=["pdf"])

        if miscarriage_pdf:
            try:
                miscarriage_inputs = extract_miscarriage_from_pdf(miscarriage_pdf, client)
                st.session_state.miscarriage_inputs = miscarriage_inputs  # ✅ store for auto-fill
                st.success("✅ Miscarriage data extracted and fields auto-filled.")
            except Exception as e:
                st.error(f"Data extraction failed: {e}")

    st.header("Manual Entry (auto-filled if PDF provided)")

    prefilled = st.session_state.get("miscarriage_inputs", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age", 15, 60, int(prefilled.get("Age", 30)))
        BMI = st.number_input("BMI", 10.0, 40.0, float(prefilled.get("BMI", 22.0)), step=0.1)
        Nmisc = st.number_input("Previous Miscarriages (Nmisc)", 0, 10, int(prefilled.get("Nmisc", 0)))
        Activity = st.selectbox("Activity Level", [0, 1], index=int(prefilled.get("Activity", 0)))
        Binking = st.selectbox("Binking", [0, 1], index=int(prefilled.get("Binking", 0)))
        Walking = st.selectbox("Walking", [0, 1], index=int(prefilled.get("Walking", 0)))
    with col2:
        Drinving = st.selectbox("Driving", [0, 1], index=int(prefilled.get("Drinving", 0)))
        Sitting = st.selectbox("Sitting", [0, 1], index=int(prefilled.get("Sitting", 0)))
        Location = st.selectbox("Location", [0, 1, 2], index=int(prefilled.get("Location", 0)))
        temp = st.number_input("Temperature (°C)", 30.0, 42.0, float(prefilled.get("temp", 37.0)), step=0.1)
        bpm = st.number_input("Heart Rate (bpm)", 40, 220, int(prefilled.get("bpm", 90)))
        stress = st.selectbox("Stress Level", [0, 1, 2, 3], index=int(prefilled.get("stress", 0)))
    with col3:
        bp = st.number_input("Blood Pressure (mmHg)", 80, 250, int(prefilled.get("bp", 120)))
        Alcohol_Consumption = st.number_input("Alcohol Consumption", 0, 1000, int(prefilled.get("Alcohol Comsumption", 0)))
        Drunk = st.selectbox("Drunk Frequency", [0, 1, 2, 3], index=int(prefilled.get("Drunk", 0)))

    miscarriage_inputs = {
        "Age": Age,
        "BMI": BMI,
        "Nmisc": Nmisc,
        "Activity": Activity,
        "Binking": Binking,
        "Walking": Walking,
        "Drinving": Drinving,
        "Sitting": Sitting,
        "Location": Location,
        "temp": temp,
        "bpm": bpm,
        "stress": stress,
        "bp": bp,
        "Alcohol Comsumption": Alcohol_Consumption,
        "Drunk": Drunk,
    }

    if st.button("Run Miscarriage Assessment"):
        try:
            miscarriage_df = pd.DataFrame([miscarriage_inputs])
            miscarriage_output = predict_miscarriage(miscarriage_model, miscarriage_df)
            query = f"Miscarriage Output: {miscarriage_output}"
            top_advices = query_docs(query, doc_embeddings, emb_model, tokenizer, advice_docs)
            report = run_risk_system_miscarriage(top_advices, miscarriage_output, client)
            render_report_dashboard(report, test_type="Miscarriage")
        except Exception as e:
            st.error(f"Error running assessment: {e}")

    if st.button("⬅ Back"):
        st.session_state.page = "intro"
        st.rerun()
