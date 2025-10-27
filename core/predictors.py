import numpy as np
from sklearn.svm import SVC

from core.embeddings import compute_embedding

# ------------------ CTG
ctg_class_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}

def load_ctg_model(model_path="models/fetal_xgb_model.json"):
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f'error loading xgboost model {e}')
        model = SVC()

    return model

def predict_ctg(model, features):
    pred = model.predict(features.reshape(1, -1))[0]
    return ctg_class_map.get(pred, "Unknown")

# ------------------ EPL

def predict_epl(MA, EM, GSD, EL, YSD, EHR):
    score = 0
    reasons = []

    if MA < 30:
        pass
    elif 30 <= MA < 35:
        score += 1
        reasons.append("Maternal age 30–34 slightly increases EPL risk.")
    elif 35 <= MA < 40:
        score += 2
        reasons.append("Maternal age 35–39 moderately increases EPL risk.")
    else:
        score += 3
        reasons.append("Maternal age ≥40 strongly increases EPL risk.")

    if EM >= 9:
        pass
    elif 7 <= EM < 9:
        score += 1
        reasons.append("Endometrium 7–8.9 mm shows borderline receptivity.")
    else:
        score += 2
        reasons.append("Endometrium <7 mm indicates poor uterine lining.")

    if GSD >= 18:
        pass
    elif 14 <= GSD < 18:
        score += 1
        reasons.append("GSD 14–17.9 mm slightly smaller than expected.")
    else:
        score += 2
        reasons.append("GSD <14 mm suggests delayed growth.")

    if EL >= 3:
        pass
    elif 1.5 <= EL < 3:
        score += 1
        reasons.append("Embryo length 1.5–2.9 mm indicates slower growth.")
    else:
        score += 2
        reasons.append("Embryo length <1.5 mm indicates poor development.")

    if not (3 <= YSD <= 4):
        score += 1
        reasons.append("Abnormal yolk sac size increases risk.")

    if EHR >= 100:
        pass
    elif 60 <= EHR < 100:
        score += 1
        reasons.append("Heart rate 60–99 bpm may indicate distress.")
    else:
        score += 2
        reasons.append("Heart rate <60 bpm predicts EPL.")
    
    prob = 1 / (1 + np.exp(-(0.35 * score - 2.5)))
    
    if score > 5:
        risk_level = "High"
    else:
        risk_level = "Low"

    return {"model": "EPL", "score": score, "risk": round(prob, 2), "risk_level": risk_level, "reasons": reasons}

# ------------------ ALL

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