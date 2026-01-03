import pandas as pd
import numpy as np

def explain_prediction(model, input_df):
    """
    Returns:
    - explanations: list[str]
    - importance_df: DataFrame(feature, importance)
    - top_features: list[str]
    """

    feature_names = input_df.columns.tolist()
    values = input_df.iloc[0]

    explanations = []
    importances = []

    for feature in feature_names:
        value = values[feature]

        if feature == "temperature" and value > 80:
            explanations.append("High temperature may cause thermal stress")
            importances.append(0.30)
        elif feature == "pressure" and value > 40:
            explanations.append("High pressure increases deviation risk")
            importances.append(0.25)
        elif feature == "process_duration" and value > 90:
            explanations.append("Extended process duration impacts quality")
            importances.append(0.20)
        elif feature == "material_quality" and value < 0.8:
            explanations.append("Low material quality reduces batch stability")
            importances.append(0.15)
        elif feature == "machine_load" and value > 75:
            explanations.append("High machine load stresses equipment")
            importances.append(0.10)
        else:
            importances.append(0.05)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top_features = importance_df.head(3)["feature"].tolist()

    if not explanations:
        explanations.append("All parameters are within normal operating ranges")

    return explanations, importance_df, top_features
