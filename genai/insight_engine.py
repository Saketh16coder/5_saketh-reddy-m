import os

def generate_batch_insight(
    prediction_label,
    prediction_score,
    top_features,
    batch_data
):
    api_key = os.getenv("OPENAI_API_KEY")

    # -------- FEATURE TEXT --------
    feature_list = [
        f[0] if isinstance(f, tuple) else f
        for f in top_features
    ]
    feature_text = ", ".join(feature_list)

    # -------- WHAT-IF LOGIC (RULE BASED) --------
    if "process_duration" in feature_list:
        what_if = (
            "If process duration increases further while other parameters remain constant, "
            "material degradation risk may increase in subsequent batches."
        )
    elif "temperature" in feature_list:
        what_if = (
            "If operating temperature drifts upward, thermal stress could accumulate and "
            "raise deviation risk over time."
        )
    else:
        what_if = (
            "If multiple parameters drift simultaneously, combined effects could "
            "increase deviation risk despite current stability."
        )

    # -------- FALLBACK GENAI (DEMO SAFE) --------
    if not api_key:
        return (
            f"**AI Risk Narrative**\n\n"
            f"This batch is classified as **{prediction_label}** with a risk score of "
            f"**{round(prediction_score, 2)}**. Key influencing parameters include "
            f"{feature_text}, all of which are currently within acceptable operating ranges.\n\n"
            f"**AI What-If Analysis**\n\n"
            f"{what_if}\n\n"
            f"**AI Confidence Note**\n\n"
            f"This insight is generated from predictive and explainable model outputs "
            f"and is intended to support, not replace, engineering judgment."
        )

    # -------- LLM POWERED GENAI --------
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""
You are a senior manufacturing quality engineer.

Batch classification: {prediction_label}
Risk score: {round(prediction_score, 2)}

Key influencing parameters:
{feature_text}

Batch parameters:
{batch_data}

Generate:
1. A short risk narrative explaining the current batch state
2. A what-if scenario describing how risk could change if conditions worsen
3. A confidence note clarifying AI's advisory role

Keep the response concise, professional, and factual.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate manufacturing risk insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return (
            f"This batch is classified as {prediction_label} with a risk score of "
            f"{round(prediction_score, 2)}. Monitoring key parameters is recommended. "
            "AI insight generation encountered a temporary issue."
        )
