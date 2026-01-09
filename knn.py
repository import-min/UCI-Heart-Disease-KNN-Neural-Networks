import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# Load dataset (expects a binary 'target' column)
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tune k using AUC
k_values = range(1, 21)
auc_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_scores.append(auc(fpr, tpr))

# Plot AUC vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, auc_scores, marker="o")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("AUC")
plt.title("KNN AUC vs k (UCI Heart Disease)")
plt.show()

best_k = k_values[np.argmax(auc_scores)]
print(f"Best k: {best_k}")

# Train final model
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(classification_report(y_test, y_pred))
