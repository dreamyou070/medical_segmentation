import torch
k_list = []
k1 = torch.randn(1,77,1280)
k_list.append(k1)
k2 = torch.randn(1,77,640)
k_list.append(k2)
k3 = torch.randn(1,77,320)
k_list.append(k3)

final_key = torch.cat(k_list, dim=2)  # 1, 77*4, dim
print(final_key.shape)