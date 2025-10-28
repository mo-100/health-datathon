import numpy as np
import tabpfn
import shap
import pandas as pd
import xgboost as xgb
import pickle

from core.embeddings import compute_embedding
from core.llm_utils import llm_generate, safe_parse_json

# ------------------ model loaders

def load_tabpfn(model_path):
    model = tabpfn.TabPFNClassifier()
    model.load_from_fit_state(model_path)
    return model

def load_xgboost(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

def load_pickle(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_ctg_model():
    model = load_pickle('ml/models/random_forest_model_stillbirth.pkl')
    return model

def load_miscarriage_model():
    model = load_pickle('ml/models/random_forest_model_miscarriage.pkl')
    return model

def load_early_fetal_loss_model():
    model = load_pickle('')
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

def predict_miscarriage(model, patient_data):
    class_map = {0: "Normal", 1: "High Risk"}
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
        elif pred_class_name == "High Risk":
            if shap_val > 0:
                recommendations.append(f"{feature} ({value}) increases risk; closer monitoring recommended.")
            else:
                recommendations.append(f"{feature} ({value}) reduces risk, but patient still at suspect level.")

    # 6️⃣ Output
    return {
        "predicted_class": pred_class_name,
        "predicted_probabilities": probs,
        "top_features": top_features,
        "recommendations": recommendations
    }

# ------------------ RISK SYSTEMS
BASE_PROMPT = f"""
You are a clinical decision support system that provides recommendations based on patient data and relevant medical literature.
Use the provided references to back up your recommendations.

OUTPUT FORMAT:
```json
{{
  "classification": "<predicted class>",
  "confidence": <predicted probability (0-100)>,
  "reason": "<brief summary of the patient's condition and reasons why this classification was made>",
  "recommendations": [
    {{
        "advice": "<advice to do (if this then do this)>",
        "source": "<(book and page)>"
    }},
    {{
        "advice": "<advice to do (if this then do this)>",
        "source": "<(book and page)>"
    }},
    {{
        "advice": "<advice to do (if this then do this)>",
        "source": "<(book and page)>"
    }}
  ]
}}
```
"""

def run_risk_system_ctg(top_advices, ctg_pred, client):
    references = "\n".join([f"{d['advice']} (Source: {d['source']}, Page: {d['page_number']})" for i, d in enumerate(top_advices)])
    input_prompt = f"Clinical summary:\nCTG Result: {ctg_pred}\nReferences:\n{references}"
    prompt = "\n".join([BASE_PROMPT, input_prompt])
    print(f"-------------------\nPROMPT:\n{prompt} ")

    llm_text = llm_generate(prompt, client)
    print(f"LLM RESPONSE:\n{llm_text}")
    json_response = safe_parse_json(llm_text)
    return json_response



def run_risk_system_miscarriage(top_advices, miscarriage_result, client):
    references = "\n".join([f"{d['advice']} (Source: {d['source']}, Page: {d['page_number']})" for i, d in enumerate(top_advices)])
    input_prompt = f"Clinical summary:\nMiscarriage Result: {miscarriage_result}\nReferences:\n{references}"
    prompt = "\n".join([BASE_PROMPT, input_prompt])
    print(f"-------------------\nPROMPT:\n{prompt} ")

    llm_text = llm_generate(prompt, client)
    print(f"LLM RESPONSE:\n{llm_text}")
    json_response = safe_parse_json(llm_text)
    return json_response
