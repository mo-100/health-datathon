import json
import numpy as np
from PyPDF2 import PdfReader

def extract_text(pdf_file):
    return "".join([page.extract_text() or "" for page in PdfReader(pdf_file).pages])

def extract_epl_from_pdf(pdf_file, client, model="qwen/qwen3-vl-30b-a3b-thinking"):
    text = extract_text(pdf_file)
    prompt = f"Extract EPL data (MA, EM, GSD, EL, YSD, EHR) from text. Return JSON. Text:\n{text}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        raise ValueError("Could not extract EPL data.")

def extract_ctg_from_pdf(pdf_file, client, model="qwen/qwen3-vl-30b-a3b-thinking"):
    text = extract_text(pdf_file)
    prompt = f"Extract CTG data as numeric JSON table with 21 features. Text:\n{text}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return np.array([data[k][0] for k in data])
    except Exception:
        raise ValueError("Could not extract CTG data.")
