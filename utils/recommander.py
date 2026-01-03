def recommend(top_features):
    actions = []

    for feature, _ in top_features:
        if feature == "temperature":
            actions.append("Reduce process temperature")
        elif feature == "pressure":
            actions.append("Lower system pressure")
        elif feature == "process_duration":
            actions.append("Optimize process duration")
        elif feature == "material_quality":
            actions.append("Perform material quality inspection")
        elif feature == "machine_load":
            actions.append("Reduce machine load")

    return actions
