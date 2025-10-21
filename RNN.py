import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
import math
import os

# --------------------------
# 1. Load the text data
# --------------------------
with open("alice.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --------------------------
# 2. Train BPE tokenizer
# --------------------------
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["alice.txt"],
    vocab_size=10000,
    min_frequency=2,
    special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
)
os.makedirs("tokenizer", exist_ok=True)
tokenizer.save_model("tokenizer")

# Load trained tokenizer
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
vocab_size = 10000

# --------------------------
# 3. Dataset preparation
# --------------------------
seq_length = 64

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=64):
        tokens = tokenizer.encode(text).ids
        self.data = []
        for i in range(len(tokens) - seq_length):
            self.data.append((tokens[i:i+seq_length], tokens[i+1:i+seq_length+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

# Split data into 90% train, 10% validation
split_idx = int(0.9 * len(text))
train_dataset = TextDataset(text[:split_idx], tokenizer, seq_length)
val_dataset = TextDataset(text[split_idx:], tokenizer, seq_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# --------------------------
# 4. Define LSTM Model
# --------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# --------------------------
# 5. Training and Evaluation
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, dataloader, val_loader=None, epochs=10, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Perplexity: {val_loss:.2f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

def evaluate(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    ppl = math.exp(total_loss / len(dataloader))
    return ppl

# --------------------------
# 6. Train and Evaluate
# --------------------------
lstm_model = LSTMLanguageModel(vocab_size)
train(lstm_model, train_loader, val_loader, epochs=10, lr=1e-3)

val_ppl = evaluate(lstm_model, val_loader)
print(f"LSTM Validation Perplexity: {val_ppl:.2f}")
