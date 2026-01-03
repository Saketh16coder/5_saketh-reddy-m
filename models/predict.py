import numpy as np

def predict_batch(input_df):
    temperature = input_df["temperature"].iloc[0]
    pressure = input_df["pressure"].iloc[0]
    process_duration = input_df["process_duration"].iloc[0]
    material_quality = input_df["material_quality"].iloc[0]
    machine_load = input_df["machine_load"].iloc[0]

    risk_score = 0
    if temperature > 80:
        risk_score += 1
    if pressure > 40:
        risk_score += 1
    if process_duration > 90:
        risk_score += 1
    if material_quality < 0.8:
        risk_score += 1
    if machine_load > 75:
        risk_score += 1

    probability = min(risk_score / 5, 1.0)

    label = 1 if risk_score >= 3 else 0
    return label, probability
