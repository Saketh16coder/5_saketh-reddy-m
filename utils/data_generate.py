import pandas as pd
import numpy as np

def generate_data(samples=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        "temperature": np.random.normal(70, 10, samples),
        "pressure": np.random.normal(30, 5, samples),
        "process_duration": np.random.normal(60, 15, samples),
        "material_quality": np.random.uniform(0.7, 1.0, samples),
        "machine_load": np.random.uniform(40, 90, samples),
    })

    data["deviation"] = (
        (data["temperature"] > 85) |
        (data["pressure"] > 40) |
        (data["process_duration"] > 90) |
        (data["material_quality"] < 0.8) |
        (data["machine_load"] > 80)
    ).astype(int)

    return data

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/raw/batch_data.csv", index=False)
