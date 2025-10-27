import streamlit as st, json, os, dotenv
from openai import Client
from core.predict_ctg import load_ctg_model
from core.extractors import extract_ctg_from_pdf, extract_epl_from_pdf
from core.embeddings import load_embedding_model, precompute_doc_embeddings
from core.run_system import run_risk_system

dotenv.load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = Client(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

st.set_page_config(page_title="EPL & CTG Risk", layout="wide")
st.title("🍼 Early Pregnancy Loss (EPL) Risk Assessment")

# Load models
ctg_model = load_ctg_model()
advice_docs = json.load(open("data/advices.jsonl"))
tokenizer, emb_model = load_embedding_model()
doc_embeddings = precompute_doc_embeddings(advice_docs, emb_model, tokenizer)

# UI inputs
col1, col2 = st.columns(2)
with col1:
    st.header("EPL Input")
    MA = st.number_input("Maternal Age", 18, 50)
    EM = st.number_input("Endometrium Thickness", 0.0, 20.0)
    GSD = st.number_input("Gestational Sac Diameter", 0.0, 50.0)
    EL = st.number_input("Embryo Length", 0.0, 10.0)
    YSD = st.number_input("Yolk Sac Diameter", 0.0, 10.0)
    EHR = st.number_input("Embryonic Heart Rate", 0, 200)
    epl_inputs = {"MA": MA, "EM": EM, "GSD": GSD, "EL": EL, "YSD": YSD, "EHR": EHR}

with col2:
    st.header("CTG Input")
    ctg_text = st.text_area("Enter 21 CTG features separated by commas")
    ctg_features = [float(x.strip()) for x in ctg_text.split(",")] if ctg_text else None

if st.button("Run Assessment"):
    if ctg_features:
        report = run_risk_system(epl_inputs, ctg_features, ctg_model, doc_embeddings, emb_model, tokenizer, advice_docs, lambda p, c: "LLM output placeholder", client)
        st.json(report)
    else:
        st.warning("Please provide CTG features.")
