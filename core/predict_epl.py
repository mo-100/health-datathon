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

    risk_percentage = min(max(score * 5, 0), 100)
    if risk_percentage < 30:
        risk_level = "Low"
    elif risk_percentage < 60:
        risk_level = "Moderate"
    elif risk_percentage < 80:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {"model": "EPL", "score": score, "risk": f"{risk_percentage}%", "risk_level": risk_level, "reasons": reasons}
