import torch
import math

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
