import numpy as np
from PyPDF2 import PdfReader

from core.llm_utils import safe_parse_json

def extract_text(pdf_file):
    return "".join([page.extract_text() or "" for page in PdfReader(pdf_file).pages])

def extract_miscarriage_from_pdf(pdf_file, client, model="qwen/qwen3-vl-30b-a3b-instruct"):
    text = extract_text(pdf_file)
    prompt = "\n".join([
        f"Extract Miscarriage data as JSON with these features:",
        "['Age', 'BMI', 'Nmisc', 'Activity', 'Binking', 'Walking', 'Drinving', 'Sitting', 'Location', 'temp', 'bpm', 'stress', 'bp', 'Alcohol Comsumption', 'Drunk', 'Miscarriage/ No Miscarriage']",
        "INPUT:\n{text}"
        ])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    try:
        return safe_parse_json(response.choices[0].message.content)
    except Exception as e:
        raise ValueError(f"Could not extract miscarriage data {e}")

def extract_ctg_from_pdf(pdf_file, client, model="qwen/qwen3-vl-30b-a3b-instruct"):
    text = extract_text(pdf_file)
    prompt = "\n".join(
        [
        "Extract CTG data as JSON with these features:",
        "[baseline_value, accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations, prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability, histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_number_of_zeroes, histogram_mode, histogram_mean, histogram_median, histogram_variance, histogram_tendency]"
        f"INPUT:\n{text}"
        ]
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    try:
        return safe_parse_json(response.choices[0].message.content)
    except Exception as e:
        raise ValueError(f"Could not extract CTG data {e}")
