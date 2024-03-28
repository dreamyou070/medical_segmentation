import torch
from safetensors import safe_open
from safetensors.torch import save_file

dir = '/share0/dreamyou070/dreamyou070/MultiSegmentation/result/medical/leader_polyp/layer_3/up_16_32_64/2_pe_basic_segmentation_model_a_cross_focal_batch_norm_head_test/position_embedder/position_embedder-000001.safetensors'
safe_open(dir, framework="pt", device="cpu")