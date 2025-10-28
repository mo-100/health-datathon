import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def compute_embedding(text, emb_model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def load_embedding_model(model_name="abhinand/MedEmbed-base-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    emb_model = AutoModel.from_pretrained(model_name)
    return tokenizer, emb_model

def precompute_doc_embeddings(docs, _emb_model, _tokenizer):
    return np.array([compute_embedding(d["advice"], _emb_model, _tokenizer) for d in docs])


def query_docs(query, doc_embeddings, emb_model, tokenizer, advice_docs, k=3):
    query_vec = compute_embedding(query, emb_model, tokenizer)
    scores = np.dot(doc_embeddings, query_vec) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_vec))
    top_docs = [advice_docs[i] for i in scores.argsort()[-k:][::-1]]
    return top_docs