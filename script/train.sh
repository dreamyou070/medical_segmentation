# !/bin/bash
# 1_pe_basic_segmentation_model_a_cross_focal_layer_norm_head
# 1_pe_basic_segmentation_model_a_cross_focal_batch_norm_head
# 1_pe_basic_segmentation_model_a_cross_focal_instance_norm_head
#
# 5_pe_concat_segmentation_model_a_cross_focal_batch_norm_head
# 6_pe_concat_segmentation_model_a_cross_focal_instance_norm_head

port_number=50004
category="medical"
obj_name="leader_polyp"
benchmark="bkai-igh-neopolyp"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="4_pe_concat_segmentation_model_a_cross_focal_layer_norm_head"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/train" \
 --test_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/test" \
 --resize_shape 512 \
 --latent_res 64 \
 --trigger_word "polyp" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --use_position_embedder --pe_do_concat \
 --n_classes 3 \
 --mask_res 256 \
 --norm_type "layer_norm"