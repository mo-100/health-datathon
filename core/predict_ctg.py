import numpy as np
import xgboost as xgb

ctg_class_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}

def load_ctg_model(model_path="models/fetal_xgb_model.json"):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

def predict_ctg(model, features):
    pred = model.predict(features.reshape(1, -1))[0]
    return ctg_class_map.get(pred, "Unknown")
