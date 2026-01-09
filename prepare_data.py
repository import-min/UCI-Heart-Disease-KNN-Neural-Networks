import pandas as pd

# Load raw dataset
df = pd.read_csv("heart_disease_uci.csv")

# Create binary target: num > 0 indicates heart disease
df["target"] = (df["num"] > 0).astype(int)

# Drop identifier and site columns
df = df.drop(columns=["num", "id", "dataset"])

# Convert boolean-like columns to numeric
bool_map = {"TRUE": 1, "FALSE": 0, True: 1, False: 0}
df["fbs"] = df["fbs"].map(bool_map)
df["exang"] = df["exang"].map(bool_map)

# One-hot encode categorical variables
categorical_cols = ["sex", "cp", "restecg", "slope", "thal"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop rows with missing values
df = df.dropna()

# Save cleaned dataset
df.to_csv("heart.csv", index=False)

print("Saved heart.csv")
print("Final shape:", df.shape)
print("Columns:", list(df.columns))

