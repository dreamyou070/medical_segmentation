import torch

original_query = torch.randn(1, 64*64, 320)
position_embedder = torch.randn(1, 64*64, 320)
# [1] concat query and position_embedder
query = torch.cat([original_query, position_embedder], dim=2) # 1,64*64,640
# [2] reshape query (dimension reduction)
# fully connected or convolution layer
query_1 = torch.nn.Linear(640, 320)(query) # 1,64*64,320
