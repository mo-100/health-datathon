import streamlit as st
import json, os, dotenv
from openai import Client
from core.extractors import extract_ctg_from_pdf, extract_epl_from_pdf
from core.embeddings import load_embedding_model, precompute_doc_embeddings, query_docs
from core.predictors import load_ctg_model, predict_epl, predict_ctg, run_risk_system_ctg, run_risk_system_epl
from core.widgets import render_report_dashboard

# ------------------ SETUP ------------------ #
dotenv.load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = Client(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

st.set_page_config(page_title="EPL & CTG Risk", layout="wide")

# ------------------ LOAD MODELS ------------------ #
ctg_model = st.cache_resource(load_ctg_model)()
with open('data/advices.jsonl') as f:
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
            st.session_state.page = "ctg" if gestational_age >= 26 else "epl"
            st.rerun()

# ------------------ PAGE 2A: CTG TEST ------------------ #
elif st.session_state.page == "ctg":
    st.title("🫀 CTG Risk Assessment")
    st.caption(f"Patient ID: {st.session_state.patient_id} | GA: {st.session_state.gestational_age} weeks")

    with st.sidebar:
        st.header("📎 Upload CTG Report (PDF)")
        ctg_pdf = st.file_uploader("Upload CTG Report (PDF)", type=["pdf"])
        ctg_features = None

        if ctg_pdf:
            try:
                ctg_features = extract_ctg_from_pdf(ctg_pdf, client)
                st.success("✅ CTG features extracted successfully.")
            except Exception as e:
                st.error(f"CTG extraction failed: {e}")

    st.header("CTG Input (Manual Entry)")
    ctg_text = st.text_area("Enter 21 CTG features separated by commas")
    if not ctg_features and ctg_text:
        ctg_features = [float(x.strip()) for x in ctg_text.split(",")]

    if st.button("Run CTG Assessment"):
        if ctg_features:
            ctg_output = predict_ctg(ctg_model, ctg_features)
            query = f"CTG Prediction: {ctg_output['predicted_class']}, top features: {ctg_output['top_features']}, recommendations: {ctg_output['recommendations']}"
            print("RAG Query:", query)
            top_advices = query_docs(query, doc_embeddings, emb_model, tokenizer, advice_docs)
            print("Top Advices:", top_advices)
            report = run_risk_system_ctg(top_advices, ctg_output, client)
            print("Report:", report)
            render_report_dashboard(report, test_type="CTG")
        else:
            st.warning("Please upload or enter CTG features.")

    if st.button("⬅ Back"):
        st.session_state.page = "intro"
        st.rerun()

# ------------------ PAGE 2B: EPL TEST ------------------ #
elif st.session_state.page == "epl":
    st.title("🍼 EPL Risk Assessment")
    st.caption(f"Patient ID: {st.session_state.patient_id} | GA: {st.session_state.gestational_age} weeks")

    with st.sidebar:
        st.header("📎 Upload EPL Ultrasound Report (PDF)")
        epl_pdf = st.file_uploader("Upload EPL Ultrasound Report (PDF)", type=["pdf"])
        epl_inputs = None

        if epl_pdf:
            try:
                epl_inputs = extract_epl_from_pdf(epl_pdf, client)
                st.success("✅ EPL data extracted successfully.")
            except Exception as e:
                st.error(f"EPL extraction failed: {e}")

    st.header("EPL Input (Manual Entry)")
    MA = st.number_input("Maternal Age", 18, 50)
    EM = st.number_input("Endometrium Thickness", 0.0, 20.0)
    GSD = st.number_input("Gestational Sac Diameter", 0.0, 50.0)
    EL = st.number_input("Embryo Length", 0.0, 10.0)
    YSD = st.number_input("Yolk Sac Diameter", 0.0, 10.0)
    EHR = st.number_input("Embryonic Heart Rate", 0, 200)
    if not epl_inputs:
        epl_inputs = {"MA": MA, "EM": EM, "GSD": GSD, "EL": EL, "YSD": YSD, "EHR": EHR}

    if st.button("Run EPL Assessment"):
        epl_output = predict_epl(**epl_inputs)
        query = f"EPL Output: {epl_output}"
        print("RAG Query:", query)
        top_advices = query_docs(query, doc_embeddings, emb_model, tokenizer, advice_docs)
        print("Top Advices:", top_advices)
        report = run_risk_system_epl(top_advices, epl_output, client)
        print("Report:", report)
        render_report_dashboard(report, test_type="EPL")


    if st.button("⬅ Back"):
        st.session_state.page = "intro"
        st.rerun()
