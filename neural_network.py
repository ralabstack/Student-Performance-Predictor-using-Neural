"""
neural_network.py
-----------------
A feedforward neural network built from scratch using NumPy only.
Architecture: Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000, random_state=42):
        np.random.seed(random_state)
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.loss_history = []

        # Xavier initialisation
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            self.z_values.append(Z)
            if i < len(self.weights) - 1:
                A = self.relu(Z)
            else:
                A = self.sigmoid(Z)
            self.activations.append(A)
        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        dA = self.activations[-1] - y
        grads_W, grads_b = [], []

        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dZ = dA
            else:
                dZ = dA * self.relu_derivative(self.z_values[i])
            dW = self.activations[i].T @ dZ / m
            db = np.mean(dZ, axis=0, keepdims=True)
            dA = dZ @ self.weights[i].T
            grads_W.insert(0, dW)
            grads_b.insert(0, db)

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    def fit(self, X, y, verbose=True):
        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            self.backward(X, y)
            if verbose and (epoch + 1) % 100 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch+1:4d}/{self.epochs} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    def predict_proba(self, X):
        return self.forward(X).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
