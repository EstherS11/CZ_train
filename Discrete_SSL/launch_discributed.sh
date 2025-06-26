#!/bin/bash

# 分布式训练启动脚本
# 使用方法: ./launch_distributed.sh [config_file] [ssl_model] [num_gpus]

# 默认参数
CONFIG=${1:-"config.yaml"}
SSL_MODEL=${2:-"wav2vec2"}  # wav2vec2, hubert, wavlm
NUM_GPUS=${3:-4}

echo "=========================================="
echo "启动分布式训练"
echo "配置文件: $CONFIG"
echo "SSL模型: $SSL_MODEL"
echo "GPU数量: $NUM_GPUS"
echo "=========================================="

# 设置环境变量
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用前4张GPU

# 方法1: 使用 torch.distributed.launch (PyTorch < 1.9)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=12355 \
#     train_distributed.py \
#     --config $CONFIG \
#     --ssl_model $SSL_MODEL \
#     --gpus $NUM_GPUS

# 方法2: 使用 torchrun (PyTorch >= 1.9, 推荐)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=12355 \
    train_distributed.py \
    --config $CONFIG \
    --ssl_model $SSL_MODEL \
    --gpus $NUM_GPUS

# 方法3: 直接运行Python脚本（脚本内部会spawn进程）
# python train_distributed.py \
#     --config $CONFIG \
#     --ssl_model $SSL_MODEL \
#     --gpus $NUM_GPUS

echo "=========================================="
echo "训练完成"
echo "=========================================="