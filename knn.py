# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc

np.random.seed(42)

# Load dataset (expects a binary 'target' column)
df = pd.read_csv("heart.csv")
print("Dataset shape:", df.shape)

X = df.drop("target", axis=1)
y = df["target"]

os.makedirs("results", exist_ok=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tune k using AUC
MAX_K = 20
k_values = range(1, MAX_K + 1)
auc_scores = []

for k in k_values:
    current_k = k
    model = KNeighborsClassifier(n_neighbors=current_k)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data = np.column_stack((fpr, tpr))
    auc_scores.append(auc(fpr, tpr))
    
np.savetxt("results/knn_last_roc_curve.csv", roc_data, delimiter=",", header="fpr,tpr", comments="")    
np.savetxt("results/knn_auc_scores.csv",
           np.column_stack((list(k_values), auc_scores)),
           delimiter=",",
           header="k,auc",
           comments="")

# Plot AUC vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, auc_scores, marker="o")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("AUC")
plt.title("KNN AUC vs k (UCI Heart Disease)")
plt.savefig("results/knn_auc_vs_k.png", dpi=300)
plt.close()

best_idx = np.argmax(auc_scores)
best_k = k_values[best_idx]
best_auc = auc_scores[best_idx]
print(f"Best k: {best_k} | AUC: {best_auc:.4f}")

with open("results/best_k.txt", "w") as f:
    f.write(str(best_k))
    
# Train final model
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
probs = final_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, probs)
roc_best = np.column_stack((fpr, tpr))
np.savetxt("results/knn_best_roc_curve.csv", roc_best, delimiter=",", header="fpr,tpr", comments="")

np.savetxt("results/knn_predictions.csv", y_pred, delimiter=",")
np.savetxt("results/knn_probabilities.csv", probs, delimiter=",")

report = classification_report(y_test, y_pred)

print(report)

with open("results/knn_classification_report.txt", "w") as f:
    f.write(report)

accuracy = (y_pred == y_test).mean()

with open("results/knn_accuracy.txt", "w") as f:
    f.write(f"{accuracy:.4f}")
