import streamlit as st
import json, os, dotenv
from openai import Client
from core.extractors import extract_ctg_from_pdf, extract_epl_from_pdf
from core.embeddings import load_embedding_model, precompute_doc_embeddings
from core.predictors import run_risk_system, load_ctg_model
from core.llm_utils import llm_generate

dotenv.load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = Client(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

st.set_page_config(page_title="EPL & CTG Risk", layout="wide")
st.title("🍼 Early Pregnancy Loss (EPL) Risk Assessment")

# Load models
ctg_model = st.cache_resource(load_ctg_model)()
with open('data/advices.jsonl') as f:
    advice_docs = json.loads(f.read())
tokenizer, emb_model = st.cache_resource(load_embedding_model)()
doc_embeddings = st.cache_resource(precompute_doc_embeddings)(advice_docs, emb_model, tokenizer)

# Sidebar for PDF upload
with st.sidebar:
    st.header("📎 Upload Reports")
    epl_pdf = st.file_uploader("Upload EPL Ultrasound Report (PDF)", type=["pdf"])
    ctg_pdf = st.file_uploader("Upload CTG Report (PDF)", type=["pdf"])

    epl_inputs = None
    ctg_features = None

    if epl_pdf:
        try:
            epl_inputs = extract_epl_from_pdf(epl_pdf, client)
            st.success("✅ EPL data extracted successfully.")
        except Exception as e:
            st.error(f"EPL extraction failed: {e}")

    if ctg_pdf:
        try:
            ctg_features = extract_ctg_from_pdf(ctg_pdf, client)
            st.success("✅ CTG features extracted successfully.")
        except Exception as e:
            st.error(f"CTG extraction failed: {e}")

# Manual input fallback
col1, col2 = st.columns(2)
with col1:
    st.header("EPL Input")
    MA = st.number_input("Maternal Age", 18, 50)
    EM = st.number_input("Endometrium Thickness", 0.0, 20.0)
    GSD = st.number_input("Gestational Sac Diameter", 0.0, 50.0)
    EL = st.number_input("Embryo Length", 0.0, 10.0)
    YSD = st.number_input("Yolk Sac Diameter", 0.0, 10.0)
    EHR = st.number_input("Embryonic Heart Rate", 0, 200)
    if not epl_inputs:
        epl_inputs = {"MA": MA, "EM": EM, "GSD": GSD, "EL": EL, "YSD": YSD, "EHR": EHR}

with col2:
    st.header("CTG Input")
    ctg_text = st.text_area("Enter 21 CTG features separated by commas")
    if not ctg_features and ctg_text:
        ctg_features = [float(x.strip()) for x in ctg_text.split(",")]

# Run assessment
if st.button("Run Assessment"):
    if ctg_features and epl_inputs:
        report = run_risk_system(epl_inputs, ctg_features, ctg_model, doc_embeddings, emb_model, tokenizer, advice_docs, llm_generate, client)
        st.subheader(f"EPL Risk: {report['EPL']['risk_level']} ({report['EPL']['risk']})")
        st.write(report["EPL"]["reasons"])
        st.subheader(f"CTG Prediction: {report['CTG']['class']}")
        st.markdown("### 🩺 Recommendations")
        st.write(report["Recommendations"])
    else:
        st.warning("Please provide both EPL and CTG inputs or PDFs.")
