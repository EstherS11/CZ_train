import os
import argparse
import yaml
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM

def download_ssl_models(config_path, model_types=['wav2vec2', 'hubert', 'wavlm']):
    """在可上网区下载所有需要的模型"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 模型配置映射
    MODEL_CONFIGS = {
        "wav2vec2": {
            "source": "facebook/wav2vec2-large",
            "model_class": Wav2Vec2,
            "vocoder": "speechbrain/hifigan-wav2vec2-k1000-LibriTTS"
        },
        "hubert": {
            "source": "facebook/hubert-large-ll60k",
            "model_class": HuBERT,
            "vocoder": "speechbrain/hifigan-hubert-k1000-LibriTTS"
        },
        "wavlm": {
            "source": "microsoft/wavlm-large",
            "model_class": WavLM,
            "vocoder": "speechbrain/hifigan-wavlm-k1000-LibriTTS"
        }
    }
    
    # 获取配置参数
    kmeans_dataset = config.get('kmeans_dataset', 'LibriSpeech960')
    num_clusters = config.get('num_clusters', 1000)
    cache_dir = config.get('cache_dir', 'pretrained_models')
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 下载每个模型
    for model_type in model_types:
        if model_type not in MODEL_CONFIGS:
            print(f"Unknown model type: {model_type}, skipping...")
            continue
            
        cfg = MODEL_CONFIGS[model_type]
        save_path = os.path.join(cache_dir, model_type)
        
        print(f"\nDownloading {model_type} components...")
        
        try:
            # 1. 下载SSL基础模型
            print(f"1. Downloading {model_type} from {cfg['source']}...")
            ssl_model = cfg["model_class"](
                source=cfg["source"],
                save_path=save_path,
                output_all_hiddens=True
            )
            print(f"✓ {model_type} base model downloaded")
            
            # 2. 下载K-means模型和vocoder
            print(f"2. Downloading K-means models for {kmeans_dataset}...")
            discrete_ssl = DiscreteSSL(
                save_path=save_path,
                ssl_model=ssl_model,
                kmeans_dataset=kmeans_dataset,
                vocoder_repo_id=cfg["vocoder"],
                num_clusters=num_clusters,
                layers_num=None  # 下载所有可用层
            )
            print(f"✓ K-means models and vocoder downloaded")
            
            # 显示可用的层
            if hasattr(discrete_ssl, 'ssl_layer_ids'):
                print(f"   Available layers: {discrete_ssl.ssl_layer_ids}")
            
            print(f"✓ Successfully downloaded all {model_type} components")
            
        except Exception as e:
            print(f"✗ Failed to download {model_type}: {e}")
            print("  Tips:")
            print("  - Check your internet connection")
            print("  - Verify kmeans_dataset name (e.g., 'LibriSpeech960')")
            print("  - Ensure you have enough disk space")
    
    print(f"\nAll models downloaded to: {os.path.abspath(cache_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--models', nargs='+', 
                       default=['wav2vec2', 'hubert', 'wavlm'],
                       help='SSL model types to download')
    args = parser.parse_args()
    
    download_ssl_models(args.config, args.models)