# !/bin/bash
# brain -> BraTS2020_Segmentation_256
# abdomen ->

# 2_absolute_pe_segmentation_model_a_cross_focal_use_batch_norm_query
# 4_absolute_pe_segmentation_model_c_cross_focal_use_batch_norm_query
# 6_absolute_pe_segmentation_model_b_cross_focal_use_batch_norm_query

port_number=51265
category="medical"
obj_name="brain"
trigger_word="brain"
benchmark="BraTS2020_Segmentation_256"
layer_name='layer_3'
sub_folder="up_16_32_64"
# --use_instance_norm
# --binary_test

accelerate launch --config_file ../../../gpu_config/gpu_0_1_config \
 --main_process_port $port_number ../train_crossattn.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/train" \
 --test_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/test" \
 --resize_shape 512 \
 --latent_res 64 \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --aggregation_model_c \
 --n_classes 4 \
 --mask_res 256 \
 --use_batchnorm \
 --use_dice_ce_loss \
 --optimizer_args weight_decay=0.00005 \
 --vae_train