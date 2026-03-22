import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Load raw dataset
df = pd.read_csv("heart_disease_uci.csv")
df.columns = df.columns.str.strip()
logging.info("Raw dataset shape: %s", df.shape)

required_cols = {"num", "fbs", "exang", "sex", "cp", "restecg", "slope", "thal"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

# Create binary target: num > 0 indicates heart disease
df["target"] = (df["num"] > 0).astype(int)

# Drop identifier and site columns
df = df.drop(columns=["num", "id", "dataset"], errors="ignore")

# Convert boolean-like columns to numeric
bool_map = {"TRUE": 1, "FALSE": 0, True: 1, False: 0}
df["fbs"] = df["fbs"].map(bool_map)
df["exang"] = df["exang"].map(bool_map)

# One-hot encode categorical variables
categorical_cols = ["sex", "cp", "restecg", "slope", "thal"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
logging.info("Number of features after encoding: %s", df.shape[1] - 1)

# Drop rows with missing values
before_rows = len(df)
df = df.dropna()
target = df.pop("target")
df["target"] = target
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
logging.info("Dropped %s rows with missing values", before_rows - len(df))

# Save cleaned dataset
df.to_csv("heart.csv", index=False)
df.head(20).to_csv("heart_preview.csv", index=False)
with open("feature_names.txt", "w") as f:
    f.write("\n".join(df.drop("target", axis=1).columns))
with open("prep_summary.txt", "w") as f:
    f.write(f"Final shape: {df.shape}\n")
    f.write(f"Target prevalence: {df['target'].mean():.4f}\n")
    f.write(f"Feature count: {df.shape[1] - 1}\n")

logging.info("Saved heart.csv")
logging.info("Final shape: %s", df.shape)
logging.info("Columns: %s", list(df.columns))
logging.info("Wrote prep_summary.txt")

if __name__ == "__main__":
    pass  # script already ran above
