import torch
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM

# 选择SSL模型类型
ssl_type = "wav2vec2"  # 可选: wav2vec2, hubert, wavlm

# SSL模型配置
model_configs = {
    "wav2vec2": {
        "source": "facebook/wav2vec2-large",  # 注意：使用large而不是base
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

config = model_configs[ssl_type]
save_path = f"./cache/{ssl_type}"

# 1. 创建SSL模型
print(f"创建 {ssl_type} 模型...")
ssl_model = config["model_class"](
    source=config["source"],
    save_path=save_path,
    output_all_hiddens=True  # 重要：必须设置为True
)
print("✓ SSL模型创建成功")

# 2. 创建DiscreteSSL包装器
print("\n创建DiscreteSSL...")
discrete_ssl = DiscreteSSL(
    save_path=save_path,
    ssl_model=ssl_model,
    kmeans_dataset="LibriSpeech960",  # 从文档表格得知
    vocoder_repo_id=config["vocoder"],  # 正确的参数名
    num_clusters=1000,
    layers_num=None  # None表示下载所有可用层
)
print("✓ DiscreteSSL创建成功")

# 查看可用的层
if hasattr(discrete_ssl, 'ssl_layer_ids'):
    print(f"可用的SSL层: {discrete_ssl.ssl_layer_ids}")

# 3. 测试推理
print("\n测试推理...")
audio = torch.randn(1, 16000)  # 1秒音频
SSL_layers = [7, 12]  # 选择一些层（从文档中的支持层选择）

# 注意：使用encode方法而不是直接调用
tokens, embeddings, processed_tokens = discrete_ssl.encode(
    audio,
    SSL_layers=SSL_layers,
    deduplicates=[False, False],
    bpe_tokenizers=[None, None]
)

print(f"✓ 推理成功!")
print(f"  Tokens shape: {tokens.shape}")
print(f"  Embeddings shape: {embeddings.shape}")
print(f"  Processed tokens shape: {processed_tokens.shape}")

# 4. 测试解码（可选）
print("\n测试解码...")
decoded_audio = discrete_ssl.decode(tokens, SSL_layers)
print(f"✓ 解码成功!")
print(f"  Decoded audio shape: {decoded_audio.shape}")