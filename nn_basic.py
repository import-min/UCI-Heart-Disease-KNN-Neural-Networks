import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, classification_report

# Load dataset
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1).values
y = df["target"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define model
class BasicNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = BasicNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = float("inf")
patience = 20
patience_counter = 0

# Train
# Train
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    current_loss = loss.item()

    # Early stopping logic
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Evaluate
with torch.no_grad():
    probs = model(X_test).cpu().numpy().ravel()
    predictions = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test.cpu().numpy(), predictions)
    auc = roc_auc_score(y_test.cpu().numpy(), probs)

print("Basic Neural Network Accuracy:", accuracy)
print("Basic Neural Network ROC-AUC:", auc)
print("\nClassification Report:\n", classification_report(y_test.cpu().numpy(), predictions, digits=3))
