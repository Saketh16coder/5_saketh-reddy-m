import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess():
    df = pd.read_csv("data/raw/batch_data.csv")

    X = df.drop("deviation", axis=1)
    y = df["deviation"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed = pd.DataFrame(X_scaled, columns=X.columns)
    processed["deviation"] = y.values

    processed.to_csv("data/processed/batch_data_processed.csv", index=False)

    return scaler

if __name__ == "__main__":
    preprocess()
