#!/bin/bash

# 设置可配置的字段
DATASET_NAME="ZhengXinXin"

# 训练 Head NeRF 模型，并记录执行时间
start_time_head=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${DATASET_NAME}/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/${DATASET_NAME}_head --reset
end_time_head=$(date +%s)
echo "Head NeRF training time: $((end_time_head - start_time_head)) seconds" | tee ${DATASET_NAME}_head_training_time.txt

# 训练 Torso NeRF 模型，并记录执行时间
start_time_torso=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${DATASET_NAME}/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/${DATASET_NAME}_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/${DATASET_NAME}_head --reset
end_time_torso=$(date +%s)
echo "Torso NeRF training time: $((end_time_torso - start_time_torso)) seconds" | tee ${DATASET_NAME}_torso_training_time.txt
