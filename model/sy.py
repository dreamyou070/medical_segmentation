import torch

q_dict = {}
k_dict = {}
q_dict[64] = [torch.randn(1,64,64,320)]

for k_res in q_dict.keys():
    query_list = q_dict[k_res]
    q_dict[k_res] = torch.cat(query_list, dim=-1)

x64_out = q_dict[64]
print(x64_out.shape)