from model.lora import create_network
from model.pe import AllPositionalEmbedding, SinglePositionalEmbedding
from model.diffusion_model import load_target_model
import os
from safetensors.torch import load_file
from model.unet import TimestepEmbedding

def call_model_package(args, weight_dtype, accelerator):


    # [1] diffusion
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)

    text_encoder.requires_grad_(False)

    vae.requires_grad_(False)
    vae.to(dtype=weight_dtype)
    vae.eval()

    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)

    # [2] lora network
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
    network = create_network(1.0, args.network_dim, args.network_alpha,
                             vae, text_encoder, unet, neuron_dropout=args.network_dropout, **net_kwargs, )
    network.apply_to(text_encoder, unet, True, True)

    unet = unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.eval()
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    if args.network_weights is not None :
        info = network.load_weights(args.network_weights)
    network.to(weight_dtype)

    # [3] PE
    position_embedder = AllPositionalEmbedding()
    if args.position_embedder_weights is not None :
        from safetensors.torch import load_file
        position_embedder_state_dict = load_file(args.position_embedder_weights)
        position_embedder.load_state_dict(position_embedder_state_dict)
        position_embedder.to(dtype=weight_dtype)

    return text_encoder, vae, unet, network, position_embedder