import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import streamlit as st

def compute_embedding(text, emb_model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

@st.cache_resource
def load_embedding_model(model_name="abhinand/MedEmbed-base-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    emb_model = AutoModel.from_pretrained(model_name)
    return tokenizer, emb_model

@st.cache_resource
def precompute_doc_embeddings(docs, emb_model, tokenizer):
    return np.array([compute_embedding(d["advice"], emb_model, tokenizer) for d in docs])
