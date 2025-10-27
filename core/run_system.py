import numpy as np, json
from .predict_epl import predict_epl
from .predict_ctg import predict_ctg
from .embeddings import compute_embedding

def run_risk_system(epl_inputs, ctg_features, ctg_model, doc_embeddings, emb_model, tokenizer, advice_docs, llm_generate, client):
    epl_result = predict_epl(**epl_inputs)
    ctg_pred = predict_ctg(ctg_model, np.array(ctg_features))

    query = f"EPL: {epl_result['risk_level']}, CTG: {ctg_pred}"
    query_vec = compute_embedding(query, emb_model, tokenizer)
    scores = np.dot(doc_embeddings, query_vec) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_vec))
    top_docs = [advice_docs[i] for i in scores.argsort()[-3:][::-1]]

    references = "\n".join([f"[{i+1}] {d['advice']} (Source: {d['source']})" for i, d in enumerate(top_docs)])
    prompt = f"Clinical summary:\nEPL Risk: {epl_result['risk']}\nCTG: {ctg_pred}\nReferences:\n{references}"

    llm_text = llm_generate(prompt, client)
    return {"EPL": epl_result, "CTG": {"class": ctg_pred}, "Recommendations": llm_text}
