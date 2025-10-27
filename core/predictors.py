import numpy as np
import tabpfn
import shap
import pandas as pd

from core.embeddings import compute_embedding
from core.llm_utils import llm_generate

# ------------------ CTG

def load_ctg_model(model_path="models/tabpfn_model.tabpfn_fit"):
    model = tabpfn.TabPFNClassifier()
    model.load_from_fit_state(model_path)
    return model



def predict_ctg(model, patient_data):
    class_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    # 1️⃣ Predict probabilities and class
    probs = model.predict_proba(patient_data)[0]
    pred_class_idx = int(np.argmax(probs))
    pred_class_name = class_map[pred_class_idx]

    # 2️⃣ Compute SHAP values for this patient
    explainer = shap.Explainer(model.predict_proba, patient_data)

    shap_values = explainer(patient_data)
    shap_vals_for_class = shap_values[:, :, pred_class_idx].values[0]

    # 3️⃣ Create a dataframe for features and SHAP values
    feature_df = pd.DataFrame({
        "feature": patient_data.columns,
        "shap_value": shap_vals_for_class,
        "feature_value": patient_data.iloc[0].values
    }).sort_values(by="shap_value", key=abs, ascending=False)

    # 4️⃣ Top 3 most important features
    top_features = feature_df.head(3)

    # 5️⃣ Generate recommendations based on SHAP values and class
    recommendations = []
    for idx, row in top_features.iterrows():
        feature, value, shap_val = row['feature'], row['feature_value'], row['shap_value']
        if pred_class_name == "Normal":
            # Positive features reassuring
            if shap_val > 0:
                recommendations.append(f"{feature} ({value}) supports normal fetal health.")
            else:
                recommendations.append(f"{feature} ({value}) slightly reduces reassurance, monitor routinely.")
        elif pred_class_name == "Suspect":
            if shap_val > 0:
                recommendations.append(f"{feature} ({value}) increases risk; closer monitoring recommended.")
            else:
                recommendations.append(f"{feature} ({value}) slightly reduces risk, but patient still at suspect level.")
        elif pred_class_name == "Pathological":
            if shap_val > 0:
                recommendations.append(f"{feature} ({value}) strongly indicates high risk; urgent monitoring/intervention required.")
            else:
                recommendations.append(f"{feature} ({value}) reduces risk but patient still at pathological level.")

    # 6️⃣ Output
    return {
        "predicted_class": pred_class_name,
        "predicted_probabilities": probs,
        "top_features": top_features,
        "recommendations": recommendations
    }

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

    return {"score": score, "risk": round(prob, 2), "risk_level": risk_level, "reasons": reasons}

# ------------------ RISK SYSTEMS
BASE_PROMPT = f"""
You are a clinical decision support system that provides recommendations based on patient data and relevant medical literature.
Use the provided references to back up your recommendations.

OUTPUT FORMAT:
```json
{{
  "classification": "<predicted class or risk level>",
  "confidence": <predicted probability or score>,
  "reason": "<brief summary of the patient's condition and reasons why this classification was made>",
  "recommendations": [
    {{
        "advice": "<recommendation>",
        "source": "<source reference>"
    }}
    ...
  ]
}}
```
"""

def run_risk_system_ctg(top_advices, ctg_pred, client):
    references = "\n".join([f"[{i+1}] {d['advice']} (Source: {d['source']})" for i, d in enumerate(top_advices)])
    prompt = f"Clinical summary:\nCTG: {ctg_pred}\nReferences:\n{references}"

    llm_text = llm_generate(prompt, client)
    return {"CTG": {"class": ctg_pred}, "Recommendations": llm_text}



def run_risk_system_epl(top_advices, epl_result, client):
    references = "\n".join([f"[{i+1}] {d['advice']} (Source: {d['source']})" for i, d in enumerate(top_advices)])
    prompt = f"Clinical summary:\nEPL Risk: {epl_result['risk']}\nReferences:\n{references}"

    llm_text = llm_generate(prompt, client)
    return {"EPL": epl_result, "Recommendations": llm_text}
