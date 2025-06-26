#!/bin/bash

# 云端环境设置脚本
# 在s11目录下运行此脚本

# 设置基础路径
BASE_PATH="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangfanrui-240108110056/s11"

echo "=========================================="
echo "设置云端训练环境"
echo "基础路径: $BASE_PATH"
echo "=========================================="

# 1. 创建必要的目录
echo "1. 创建目录结构..."
mkdir -p $BASE_PATH/json_files
mkdir -p $BASE_PATH/pretrained_models
mkdir -p $BASE_PATH/experiments
mkdir -p $BASE_PATH/code

# 2. 提示用户上传JSON文件
echo ""
echo "2. 请确保以下步骤已完成:"
echo "   a) 在本地运行: python convert_paths.py --verify"
echo "   b) 将生成的cloud_json/目录下的文件上传到: $BASE_PATH/json_files/"
echo ""

# 3. 检查音频文件
echo "3. 检查音频文件..."
AUDIO_PATH="$BASE_PATH/DATA/Audios"
if [ -d "$AUDIO_PATH" ]; then
    AUDIO_COUNT=$(ls -1 $AUDIO_PATH/*.wav 2>/dev/null | wc -l)
    echo "   找到 $AUDIO_COUNT 个音频文件"
else
    echo "   错误: 音频目录不存在!"
    exit 1
fi

# 4. 创建环境变量文件
echo ""
echo "4. 创建环境变量文件..."
cat > $BASE_PATH/code/env_vars.sh << EOF
#!/bin/bash
# 环境变量设置

export BASE_PATH="$BASE_PATH"
export DATA_PATH="\$BASE_PATH/DATA/Audios"
export JSON_PATH="\$BASE_PATH/json_files"
export MODEL_PATH="\$BASE_PATH/pretrained_models"
export EXP_PATH="\$BASE_PATH/experiments"

# CUDA设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

echo "环境变量已设置:"
echo "  BASE_PATH: \$BASE_PATH"
echo "  DATA_PATH: \$DATA_PATH"
echo "  JSON_PATH: \$JSON_PATH"
echo "  MODEL_PATH: \$MODEL_PATH"
echo "  EXP_PATH: \$EXP_PATH"
EOF

chmod +x $BASE_PATH/code/env_vars.sh

# 5. 创建快速启动脚本
echo ""
echo "5. 创建快速启动脚本..."
cat > $BASE_PATH/code/run_training.sh << EOF
#!/bin/bash
# 快速启动训练脚本

# 加载环境变量
source $BASE_PATH/code/env_vars.sh

# 选择SSL模型类型
SSL_MODEL=\${1:-"wav2vec2"}  # 默认使用wav2vec2

echo "开始训练 - SSL模型: \$SSL_MODEL"

# 切换到代码目录
cd $BASE_PATH/code

# 下载预训练模型（如果需要）
if [ ! -d "\$MODEL_PATH/\$SSL_MODEL" ]; then
    echo "下载预训练模型..."
    python download_models.py --config config_cloud.yaml --models \$SSL_MODEL
fi

# 启动分布式训练
torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node=4 \\
    --master_port=12355 \\
    train_distributed.py \\
    --config config_cloud.yaml \\
    --ssl_model \$SSL_MODEL \\
    --gpus 4
EOF

chmod +x $BASE_PATH/code/run_training.sh

# 6. 创建数据验证脚本
echo ""
echo "6. 创建数据验证脚本..."
cat > $BASE_PATH/code/verify_data.py << EOF
import json
import os
import sys

def verify_json_and_audio(json_path, base_path):
    """验证JSON文件中的路径是否都存在对应的音频文件"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    found = 0
    missing = []
    
    for utt_id, item in data.items():
        wav_path = item['wav']
        if os.path.exists(wav_path):
            found += 1
        else:
            missing.append(wav_path)
    
    print(f"文件: {os.path.basename(json_path)}")
    print(f"  总条目: {total}")
    print(f"  找到文件: {found}")
    print(f"  缺失文件: {total - found}")
    
    if missing:
        print(f"  缺失的前5个文件:")
        for i, path in enumerate(missing[:5]):
            print(f"    {path}")
    
    return found == total

if __name__ == "__main__":
    base_path = "$BASE_PATH"
    json_dir = os.path.join(base_path, "json_files")
    
    print("验证数据完整性...")
    print("-" * 50)
    
    all_valid = True
    for json_file in ['msp_train_10class.json', 'msp_valid_10class.json', 'msp_test_10class.json']:
        json_path = os.path.join(json_dir, json_file)
        if os.path.exists(json_path):
            valid = verify_json_and_audio(json_path, base_path)
            all_valid = all_valid and valid
        else:
            print(f"警告: JSON文件不存在 - {json_path}")
            all_valid = False
        print()
    
    if all_valid:
        print("✓ 所有数据验证通过！")
    else:
        print("✗ 数据验证失败，请检查文件路径")
        sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "设置完成！"
echo ""
echo "下一步操作:"
echo "1. 在本地转换JSON文件路径:"
echo "   python convert_paths.py --verify"
echo ""
echo "2. 上传转换后的JSON文件到云端:"
echo "   scp cloud_json/*.json user@server:$BASE_PATH/json_files/"
echo ""
echo "3. 将代码文件上传到云端:"
echo "   scp *.py config_cloud.yaml user@server:$BASE_PATH/code/"
echo ""
echo "4. 验证数据:"
echo "   cd $BASE_PATH/code && python verify_data.py"
echo ""
echo "5. 开始训练:"
echo "   cd $BASE_PATH/code && ./run_training.sh wav2vec2"
echo "=========================================="