import numpy as np
import pandas as pd
from q4 import Linear, ReLU, CrossEntropy  


train = pd.read_csv("train.csv", header=None)
Y = train.iloc[:4000, 0].values.astype(int)
X = train.iloc[:4000, 1:].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Transpose
X = X.T
num_samples = X.shape[1]
input_dim = X.shape[0]

num_classes = len(np.unique(Y))
y_onehot = np.eye(num_classes)[Y].T

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, num_classes):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, num_classes)
        self.criterion = CrossEntropy()

    def forward(self, x, y, debug=False):
        out = self.fc1.forward(x)
        out = self.relu.forward(out)
        logits = self.fc2.forward(out)
        loss = self.criterion.forward(logits, y)
        
        if debug:
            print("fc1 mean/std:", out.mean(), out.std())
            print("logits mean/std:", logits.mean(), logits.std())
        
        return logits, loss

    def backward(self, lr=0.1):
        grad = self.criterion.backward()
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)

        # Update weights
        self.fc2.W -= lr * self.fc2.dW
        self.fc2.b -= lr * self.fc2.db
        self.fc1.W -= lr * self.fc1.dW
        self.fc1.b -= lr * self.fc1.db


hidden_dim = 128
model = SimpleNN(input_dim, hidden_dim, num_classes)

epochs = 200
lr = 0.1
batch_size = 64

for epoch in range(epochs):
    # Shuffle data
    perm = np.random.permutation(num_samples)
    X_shuffled = X[:, perm]
    y_shuffled = y_onehot[:, perm]

    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        y_batch = y_shuffled[:, i:i+batch_size]

        logits, loss = model.forward(X_batch, y_batch)
        model.backward(lr=lr)

    # Evaluate full dataset each epoch
    logits_full, loss_full = model.forward(X, y_onehot)
    preds = np.argmax(logits_full, axis=0)
    acc = (preds == Y).mean()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss_full:.4f}, Accuracy: {acc:.4f}")

