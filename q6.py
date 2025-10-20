import numpy as np
import pandas as pd
from q4 import Linear, ReLU, CrossEntropy  
import matplotlib.pyplot as plt

# Load training data
train = pd.read_csv("train.csv", header=None)
Y_train = train.iloc[:4000, 0].values.astype(int)  
X_train = train.iloc[:4000, 1:].values            

# Normalize features
X_train_mean = X_train.mean(axis=0, keepdims=True)
X_train_std = X_train.std(axis=0, keepdims=True)
X_train = (X_train - X_train_mean) / X_train_std
X_train = X_train.T
num_samples = X_train.shape[1]
input_dim = X_train.shape[0]

num_classes = len(np.unique(Y_train))
y_train_onehot = np.eye(num_classes)[Y_train].T

#LOAD Validation data 4000-8500
X_val = train.iloc[4000:, 1:].values
Y_val = train.iloc[4000:, 0].values.astype(int)
X_val = (X_val - X_train_mean) / X_train_std
X_val = X_val.T
y_val_onehot = np.eye(num_classes)[Y_val].T

# Load test data
test = pd.read_csv("test.csv", header=None)
Y_test = test.iloc[:, 0].values.astype(int)
X_test = test.iloc[:, 1:].values
X_test = (X_test - X_train_mean) / X_train_std
X_test = X_test.T
y_test_onehot = np.eye(num_classes)[Y_test].T

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, num_classes):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, num_classes)
        self.criterion = CrossEntropy()

    def forward(self, x, y):
        out = self.fc1.forward(x)
        out = self.relu.forward(out)
        logits = self.fc2.forward(out)
        loss = self.criterion.forward(logits, y)
        return logits, loss

    def backward(self, lr=0.1, weight_decay=1e-4):
        grad = self.criterion.backward()
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)

        # Update weights with L2 regularization
        self.fc2.W -= lr * (self.fc2.dW + weight_decay * self.fc2.W)
        self.fc2.b -= lr * self.fc2.db
        self.fc1.W -= lr * (self.fc1.dW + weight_decay * self.fc1.W)
        self.fc1.b -= lr * self.fc1.db

# ------------------------------------------------------------- 
#UNCOMMENT THIS TO RUN HYPERPARAMETER TUNING!!
# # Helper function to train and evaluate
# def train_nn(hidden_dim=128, lr=0.1, batch_size=64, weight_decay=0.0, epochs=50):
#     model = SimpleNN(input_dim, hidden_dim, num_classes)
#     for epoch in range(epochs):
#         perm = np.random.permutation(num_samples)
#         X_shuffled = X[:, perm]
#         y_shuffled = y_onehot[:, perm]

#         for i in range(0, num_samples, batch_size):
#             X_batch = X_shuffled[:, i:i+batch_size]
#             y_batch = y_shuffled[:, i:i+batch_size]
#             logits, loss = model.forward(X_batch, y_batch)
#             model.backward(lr=lr, weight_decay=weight_decay)

#     logits_full, _ = model.forward(X, y_onehot)
#     preds = np.argmax(logits_full, axis=0)
#     acc = (preds == Y).mean()
#     return acc

# #Accuracy vs Hidden Units
# hidden_units_list = [32, 64, 128, 256]
# acc_hidden_units = [train_nn(hidden_dim=h, lr=0.1, batch_size=64, weight_decay=1e-4, epochs=50) 
#                     for h in hidden_units_list]

# # Accuracy vs Learning Rate
# lr_list = [0.01, 0.05, 0.1, 0.3]
# acc_lr = [train_nn(hidden_dim=128, lr=lr_val, batch_size=64, weight_decay=1e-4, epochs=50) 
#           for lr_val in lr_list]

# # Accuracy vs Batch Size
# batch_sizes = [32, 64, 128]
# acc_batch = [train_nn(hidden_dim=128, lr=0.1, batch_size=bs, weight_decay=1e-4, epochs=50) 
#              for bs in batch_sizes]

# #Accuracy vs Weight Decay (L2)
# weight_decay_list = [0, 1e-5, 1e-4, 1e-3]
# acc_wd = [train_nn(hidden_dim=128, lr=0.1, batch_size=64, weight_decay=wd, epochs=50) 
#           for wd in weight_decay_list]

# # graphhs :)
# plt.figure(figsize=(12, 10))

# plt.subplot(2, 2, 1)
# plt.plot(hidden_units_list, acc_hidden_units, marker='o')
# plt.title("Accuracy vs Hidden Units")
# plt.xlabel("Hidden Units")
# plt.ylabel("Accuracy")
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(lr_list, acc_lr, marker='o', color='orange')
# plt.title("Accuracy vs Learning Rate")
# plt.xlabel("Learning Rate")
# plt.ylabel("Accuracy")
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(batch_sizes, acc_batch, marker='o', color='green')
# plt.title("Accuracy vs Batch Size")
# plt.xlabel("Batch Size")
# plt.ylabel("Accuracy")
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(weight_decay_list, acc_wd, marker='o', color='red')
# plt.title("Accuracy vs Weight Decay (L2)")
# plt.xlabel("Weight Decay")
# plt.ylabel("Accuracy")
# plt.grid(True)

# plt.tight_layout()
# plt.show()

#------------------------------------------------------------ DONT UNCOMMENT ABOVE

# ---------------------------------------------------------- COMMENT THIS IF RUNNING HYPERPARAMETER TUNING !! UNCOMMENT IF NOT!!
#chosen hyperparameters from graphs 
# hidden_dim=256, lr=0.3, batch_size=32, weight_decay=1e-4

# Training hyperparameters
hidden_dim = 32
lr = 0.3
batch_size = 32
weight_decay = 1e-4
epochs = 50

model = SimpleNN(input_dim, hidden_dim, num_classes)

# Store training and validation accuracy for plotting
train_acc_history = []
val_acc_history = []
train_loss_history = []  

for epoch in range(epochs):
    perm = np.random.permutation(num_samples)
    X_shuffled = X_train[:, perm]
    y_shuffled = y_train_onehot[:, perm]

    epoch_loss = 0.0  
    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        y_batch = y_shuffled[:, i:i+batch_size]
        logits, loss = model.forward(X_batch, y_batch)
        model.backward(lr=lr, weight_decay=weight_decay)
        epoch_loss += loss

    epoch_loss /= (num_samples / batch_size)
    train_loss_history.append(epoch_loss)

    # Compute training accuracy
    logits_train, _ = model.forward(X_train, y_train_onehot)
    preds_train = np.argmax(logits_train, axis=0)
    acc_train = (preds_train == Y_train).mean()
    train_acc_history.append(acc_train)

    # Compute validation accuracy
    logits_val, _ = model.forward(X_val, y_val_onehot)
    preds_val = np.argmax(logits_val, axis=0)
    acc_val = (preds_val == Y_val).mean()
    val_acc_history.append(acc_val)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}, Training Accuracy: {acc_train:.4f}, Validation Accuracy: {acc_val:.4f}")


# Evaluate on test set
logits_test, _ = model.forward(X_test, y_test_onehot)
preds_test = np.argmax(logits_test, axis=0)
test_acc = (preds_test == Y_test).mean()
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Plot training vs validation accuracy and loss
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_acc_history, label='Training Accuracy')
plt.plot(range(1, epochs+1), val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_loss_history, label='Training Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
