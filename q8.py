import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

# Load training data
train = pd.read_csv("train.csv", header=None)

# Training set: first 4000 samples
X_train = train.iloc[:4000, 1:].values.astype("float32")
Y_train = train.iloc[:4000, 0].values.astype(int)

# Validation set: remaining rows
X_val = train.iloc[4000:, 1:].values.astype("float32")
Y_val = train.iloc[4000:, 0].values.astype(int)

# Normalize features 
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True)
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

# Convert to tensors
X_train_tensor = torch.tensor(X_train)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

# Hyperparameters -> these weere the same as q6.py
input_dim = X_train.shape[1]
hidden_dim = 32
num_classes = len(set(Y_train))
batch_size = 32
lr = 0.3
weight_decay = 1e-4
epochs = 50

# Dataloader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN(input_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training loop with loss tracking
train_acc_history = []
val_acc_history = []
train_loss_history = []

for epoch in range(epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += Y_batch.size(0)
        correct_train += (predicted == Y_batch).sum().item()
    
    # Average training loss
    avg_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_loss)
    
    # Training accuracy
    train_acc = correct_train / total_train
    train_acc_history.append(train_acc)

    # Validation accuracy
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor).item()
        _, val_predicted = torch.max(val_outputs, 1)
        val_acc = (val_predicted == Y_val_tensor).sum().item() / Y_val_tensor.size(0)
        val_acc_history.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Evaluate on test set
test = pd.read_csv("test.csv", header=None)
X_test = test.iloc[:, 1:].values.astype("float32")
Y_test = test.iloc[:, 0].values.astype(int)
X_test = (X_test - X_mean) / X_std
X_test_tensor = torch.tensor(X_test)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs, 1)
    test_acc = (test_predicted == Y_test_tensor).sum().item() / Y_test_tensor.size(0)

print(f"\nFinal Test Accuracy: {test_acc:.4f}")


# Plot convergence curves
plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), train_loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Convergence")
plt.grid(True)
plt.legend()
plt.show()


# note that I have answered these questions in my paper SML REPORT.pdf But here are my answers again:

#Compare the results (e.g., training loss, validation accuracy, convergence speed) with those from your custom implementation.
# The training loss curve shows a steep decrease over epochs, indicating not only effective but fast learning. This is slightly better than my custom implementation where the loss curve was less steep and more gradual.
# The validation accuracy improves consistently, suggesting good generalization, remanining consistently at 98percent accuraccy which was higher than my implementation of 96 percent.
# In addition, there is less overfitting as the pytorch implentation has a smaller gap between training and validation accuracy.
# The convergence speed is faster than my implementation, as can be seen from the sharper drop in training loss.
#the similarities are that both implementations show a decrease in training loss and an increase in validation accuracy over epochs, indicating effective learning.
# Both implementations also achieve high accuracy on the validation set, demonstrating good generalization to unseen data.
