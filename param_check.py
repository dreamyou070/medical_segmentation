import os
from model.unet import UNet2DConditionModel
unet = UNet2DConditionModel(sample_size = 512)


def register_optimizer_param(net_, layer_name, trainable_params):
    if net_.__class__.__name__ == 'CrossAttention':



        trainable_params.append(net_.parameters())
    elif hasattr(net_, 'children'):
        for name__, net__ in net_.named_children():
            full_name = f'{layer_name}_{name__}'
            count = register_optimizer_param(net__, full_name, trainable_params)
    return trainable_params

params = []
for net in unet.named_children():
    if "down" in net[0]:
        params = register_optimizer_param(net[1], net[0], trainable_params=params)
    elif "up" in net[0]:
        params = register_optimizer_param(net[1], net[0], trainable_params=params)
    elif "mid" in net[0]:
        params = register_optimizer_param(net[1], net[0], trainable_params=params)
trainable_params = [{"params": params, "lr": 0.001}]
print(trainable_params)