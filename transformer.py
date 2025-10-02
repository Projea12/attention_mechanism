import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# 1. Toy dataset: counting sequences

def make_dataset(seq_len=5, num_samples=5000, vocab_size=20):
    X, Y = [], []
    for _ in range(num_samples):
        start = random.randint(0, vocab_size - seq_len - 1)
        seq = list(range(start, start + seq_len))
        target = start + seq_len  # next number
        X.append(seq)
        Y.append(target)
    return torch.tensor(X), torch.tensor(Y)

X, Y = make_dataset()
vocab_size = 50
seq_len = X.size(1)

# 2. Tiny Transformer Model

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer(x)
        # use only the last token for prediction
        x = x[:, -1, :]
        return self.fc_out(x)

model = TinyTransformer(vocab_size=vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 3. Training Loop

for epoch in range(50):  # to get better result increase the Epoch
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4. Test the model

test_seq = torch.tensor([[5,6,7,8,9]])
pred = model(test_seq).argmax(dim=1).item()
print("Input:", test_seq.tolist(), "Prediction:", pred)
