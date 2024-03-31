import torch

query = torch.randn(1, 64, 1280)
key = torch.randn(1, 77, 1280)
attention_scores = torch.matmul(query, key.transpose(-1, -2),)
print(attention_scores.shape)