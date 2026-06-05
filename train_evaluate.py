"""
train_evaluate.py
-----------------
Full training pipeline: load data, preprocess, train, evaluate, and save results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from neural_network import NeuralNetwork

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('student_data.csv')
print("Dataset shape:", df.shape)
print(df.head())

X = df.drop('passed', axis=1).values
y = df['passed'].values

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 3. Normalise ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. Build & Train Network ──────────────────────────────────────────────────
# Architecture: 8 inputs -> 16 -> 8 -> 1 output
nn = NeuralNetwork(layer_sizes=[8, 16, 8, 1], learning_rate=0.05, epochs=500)
print("\nTraining...\n")
nn.fit(X_train, y_train, verbose=True)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
train_acc = nn.accuracy(X_train, y_train)
test_acc  = nn.accuracy(X_test, y_test)
print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Test  Accuracy: {test_acc*100:.2f}%")

y_pred = nn.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Failed", "Passed"]))

# ── 6. Plot: Loss Curve ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Student Performance Predictor — Neural Network", fontsize=14, fontweight='bold')

axes[0].plot(nn.loss_history, color='steelblue', linewidth=1.5)
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].grid(True, alpha=0.3)

# ── 7. Plot: Confusion Matrix ─────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_title("Confusion Matrix (Test Set)")
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Predicted: Fail', 'Predicted: Pass'])
axes[1].set_yticklabels(['Actual: Fail', 'Actual: Pass'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=18, color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as results.png")
