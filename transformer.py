import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import ByteLevelBPETokenizer
import math
import os

# --------------------------
# 1. Load the text data
# --------------------------
with open("alice.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --------------------------
# 2. Train a fresh BPE tokenizer
# --------------------------
os.makedirs("tokenizer", exist_ok=True)
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["alice.txt"],
    vocab_size=10000,
    min_frequency=2,
    special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
)
tokenizer.save_model("tokenizer")

# Reload for encoding
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
vocab_size = tokenizer.get_vocab_size()
print(f"✅ New tokenizer trained. Vocabulary size: {vocab_size}")

# --------------------------
# 3. Dataset preparation
# --------------------------
seq_length = 64

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        tokens = tokenizer.encode(text).ids
        self.data = []
        for i in range(len(tokens) - seq_length):
            self.data.append((tokens[i:i+seq_length], tokens[i+1:i+seq_length+1]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

dataset = TextDataset(text, tokenizer, seq_length)

# Split 90% train / 10% validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# --------------------------
# 4. Define Transformer Model
# --------------------------
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=2, dim_feedforward=512, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_seq_len, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # avoids transpose
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        x = self.transformer(x)  # (batch, seq_len, embed)
        out = self.fc(x)
        return out

# --------------------------
# 5. Training function
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_loader, val_loader, epochs=5, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Perplexity: {val_ppl:.2f}")

# --------------------------
# 6. Train and Evaluate
# --------------------------
transformer_model = TransformerLanguageModel(vocab_size)
train(transformer_model, train_loader, val_loader, epochs=10, lr=1e-3)
