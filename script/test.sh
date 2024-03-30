# !/bin/bash

port_number=50042
category="medical"
obj_name="brain"
benchmark="BraTS2020_Segmentation_128"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="4_absolute_pe_segmentation_model_c_cross_focal_use_batch_norm_query"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../test.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/test" \
 --network_folder "../../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}/model" \
 --obj_name "${obj_name}" \
 --prompt "${trigger_word}" \
 --latent_res 64 \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --use_position_embedder \
 --aggregation_model_c \
 --n_classes 4 \
 --mask_res 128 \
 --use_batchnorm