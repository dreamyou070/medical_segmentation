import torch
deactivating_loss = []
loss1 = torch.randn((5,6))
loss2 = torch.randn((5,6))
deactivating_loss.append(loss1)
deactivating_loss.append(loss2)
deactivating_loss = torch.stack(deactivating_loss).sum()
print(deactivating_loss)