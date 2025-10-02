import torch
import math
import torch.nn as nn

def baby_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn

q = torch.randn(1, 3, 4) 
k = torch.randn(1, 3, 4)
v = torch.randn(1, 3, 4)

out, attn = baby_attention(q, k, v)
print("out shape:", out.shape)
print("attn shape:", attn.shape)


def baby_attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn




class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        out, attn = baby_attention(q, k, v, mask=mask)
        return self.out(out), attn

