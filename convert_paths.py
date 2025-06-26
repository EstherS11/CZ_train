import json
import os
import argparse
from pathlib import Path

def convert_json_paths(json_file, output_file, cloud_base_path):
    """
    转换JSON文件中的本地路径到云端路径
    
    Args:
        json_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        cloud_base_path: 云端的基础路径
    """
    # 读取原始JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 统计信息
    total_count = len(data)
    converted_count = 0
    
    # 转换每个条目的路径
    for utt_id, item in data.items():
        if 'wav' in item:
            # 提取文件名
            original_path = item['wav']
            filename = os.path.basename(original_path)
            
            # 构建新的云端路径
            new_path = os.path.join(cloud_base_path, filename)
            
            # 更新路径
            item['wav'] = new_path
            converted_count += 1
    
    # 保存转换后的JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"转换完成: {json_file} -> {output_file}")
    print(f"总条目数: {total_count}, 转换数: {converted_count}")
    
    return data


def batch_convert_json_files(input_dir, output_dir, cloud_base_path):
    """
    批量转换目录下的所有JSON文件
    
    Args:
        input_dir: 包含原始JSON文件的目录
        output_dir: 输出目录
        cloud_base_path: 云端的基础路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 需要转换的JSON文件列表
    json_files = [
        'msp_train_10class.json',
        'msp_valid_10class.json', 
        'msp_test_10class.json'
    ]
    
    print(f"云端基础路径: {cloud_base_path}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 转换每个文件
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        
        if os.path.exists(input_path):
            convert_json_paths(input_path, output_path, cloud_base_path)
        else:
            print(f"警告: 文件不存在 - {input_path}")
    
    print("-" * 50)
    print("所有文件转换完成！")


def verify_cloud_paths(json_file, max_samples=5):
    """
    验证转换后的路径
    
    Args:
        json_file: 转换后的JSON文件
        max_samples: 显示的样本数量
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n验证文件: {json_file}")
    print(f"显示前{max_samples}个样本:")
    
    for i, (utt_id, item) in enumerate(data.items()):
        if i >= max_samples:
            break
        print(f"  {utt_id}: {item['wav']}")
    
    print(f"总共 {len(data)} 个条目\n")


def main():
    parser = argparse.ArgumentParser(description='转换MSP-PODCAST JSON文件的路径')
    parser.add_argument('--input_dir', type=str, default='.',
                       help='包含原始JSON文件的目录 (默认: 当前目录)')
    parser.add_argument('--output_dir', type=str, default='cloud_json',
                       help='输出目录 (默认: cloud_json)')
    parser.add_argument('--cloud_path', type=str, 
                       default='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangfanrui-240108110056/s11/DATA/Audios',
                       help='云端音频文件的基础路径')
    parser.add_argument('--verify', action='store_true',
                       help='转换后验证路径')
    
    args = parser.parse_args()
    
    # 执行批量转换
    batch_convert_json_files(args.input_dir, args.output_dir, args.cloud_path)
    
    # 如果需要验证
    if args.verify:
        print("\n" + "="*50)
        print("验证转换后的路径:")
        print("="*50)
        
        json_files = ['msp_train_10class.json', 'msp_valid_10class.json', 'msp_test_10class.json']
        for json_file in json_files:
            output_path = os.path.join(args.output_dir, json_file)
            if os.path.exists(output_path):
                verify_cloud_paths(output_path, max_samples=3)


if __name__ == '__main__':
    main()