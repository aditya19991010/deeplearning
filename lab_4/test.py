import pandas as pd
import numpy as np

# ---- DATA PREP ----
data = [
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 0]
]
columns = ['x1', 'x2', 'x3', 'y']
df = pd.DataFrame(data, columns=columns)

# INPUTS and LABELS
X = df.iloc[:, :-1].values  # shape (4, 3)
y = df.iloc[:, -1].values.reshape(-1, 1)  # shape (4, 1)

# ---- HYPERPARAMETERS ----
epochs = 1000
lr = 0.1
n_hidden = 5
n_output = 1
n_input = X.shape[1]

# ---- WEIGHT INITIALIZATION ----
W1 = np.random.randn(n_input, n_hidden)
b1 = np.zeros((1, n_hidden))

W2 = np.random.randn(n_hidden, n_output)
b2 = np.zeros((1, n_output))

# ---- ACTIVATION FUNCTIONS ----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# ---- LOSS FUNCTION ----
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ---- TRAINING LOOP ----
for epoch in range(epochs):
    # ---- FORWARD PASS ----
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)  # Output

    # ---- LOSS ----
    loss = mse(y, A2)

    # ---- BACKPROPAGATION ----
    dZ2 = (A2 - y) * sigmoid_derivative(Z2)  # shape (4,1)
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # ---- WEIGHT UPDATE ----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # ---- LOGGING ----
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# ---- PREDICTION ----
predictions = (A2 > 0.5).astype(int)
print("\nFinal predictions:", predictions.ravel())
