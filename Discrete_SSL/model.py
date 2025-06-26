import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
import os

# ===================================================================
# =============         ECAPA-TDNN COMPONENTS          =============
# ===================================================================

class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return x * self.se(x)


class Res2Block(nn.Module):
    """Res2Net block with SE."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, scale=4):
        super().__init__()
        self.scale = scale
        width = out_channels // scale
        
        self.conv1 = nn.Conv1d(in_channels, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=kernel_size, 
                     dilation=dilation, padding=(kernel_size-1)*dilation//2)
            for _ in range(scale-1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale-1)])
        
        self.conv3 = nn.Conv1d(width * scale, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.se = SEModule(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                       if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        xs = torch.chunk(x, self.scale, dim=1)
        ys = []
        for i in range(self.scale):
            if i == 0:
                ys.append(xs[i])
            elif i == 1:
                ys.append(F.relu(self.bns[i-1](self.convs[i-1](xs[i]))))
            else:
                ys.append(F.relu(self.bns[i-1](self.convs[i-1](xs[i] + ys[-1]))))
        
        x = torch.cat(ys, dim=1)
        x = self.bn3(self.conv3(x))
        x = self.se(x)
        return F.relu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling."""
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x (torch.Tensor): [batch, channels, time]
            lengths (torch.Tensor, optional): [batch] actual sequence lengths (NOT ratios). Defaults to None.
        """
        alpha = self.attention(x)
        if lengths is not None:
            batch_size, _, time_dim = x.shape
            mask = torch.arange(time_dim, device=x.device).expand(batch_size, time_dim)
            mask = mask < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
            alpha = alpha * mask
            alpha = alpha / (alpha.sum(dim=2, keepdim=True) + 1e-8)
        
        mean = torch.sum(alpha * x, dim=2)
        std = torch.sqrt(torch.sum(alpha * (x - mean.unsqueeze(2)) ** 2, dim=2).clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    """Full ECAPA-TDNN implementation."""
    def __init__(self, input_dim=256, channels=[512, 512, 512, 512, 1536], 
                 kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                 attention_channels=128, lin_neurons=192):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, channels[0], kernel_size=kernel_sizes[0], padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels[0])
        
        self.blocks = nn.ModuleList([
            Res2Block(channels[i], channels[i+1], kernel_sizes[i+1], dilations[i+1])
            for i in range(len(channels)-1)
        ])
        
        self.mfa = nn.Conv1d(sum(channels), channels[-1], kernel_size=1)
        self.pooling = AttentiveStatisticsPooling(channels[-1], attention_channels)
        
        self.fc1 = nn.Linear(channels[-1] * 2, lin_neurons)
        self.bn2 = nn.BatchNorm1d(lin_neurons)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x (torch.Tensor): [batch, input_dim, time]
            lengths (torch.Tensor, optional): [batch] actual sequence lengths in time dimension. Defaults to None.
        """
        x = self.bn1(self.relu1(self.conv1(x)))
        layer_outputs = [x]
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)
        
        x = torch.cat(layer_outputs, dim=1)
        x = self.mfa(x)
        x = self.pooling(x, lengths)
        x = self.dropout(self.relu2(self.bn2(self.fc1(x))))
        return x


# ===================================================================
# =============     DISCRETE SSL COMPONENTS         =============
# ===================================================================

class DiscreteEmbeddingLayer(nn.Module):
    """Embedding layer for discrete SSL tokens with explicit layer mapping."""
    
    def __init__(self, num_clusters, embedding_dim, ssl_layers):
        super().__init__()
        self.num_clusters = num_clusters
        self.ssl_layers = ssl_layers
        
        self.embeddings = nn.ModuleDict({
            str(layer_id): nn.Embedding(num_clusters, embedding_dim)
            for layer_id in self.ssl_layers
        })

    def forward(self, tokens):
        """
        Args:
            tokens (torch.Tensor): [batch, time, num_selected_layers]
        Returns:
            torch.Tensor: [batch, time, num_selected_layers, embedding_dim]
        """
        batch_size, seq_len, num_selected_layers = tokens.shape
        assert num_selected_layers == len(self.ssl_layers), \
            f"Number of token layers ({num_selected_layers}) does not match number of embedding layers ({len(self.ssl_layers)})"
            
        embeddings_list = []
        for i, layer_id in enumerate(self.ssl_layers):
            layer_tokens = tokens[:, :, i]
            embedding_layer = self.embeddings[str(layer_id)]
            emb = embedding_layer(layer_tokens)
            embeddings_list.append(emb)
            
        return torch.stack(embeddings_list, dim=2)


class AttentionPooling(nn.Module):
    """Attention-based pooling for discrete tokens."""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): [batch, time, dim]
            mask (torch.Tensor, optional): [batch, time] binary mask. Defaults to None.
        Returns:
            tuple: (pooled output [batch, dim], attention weights [batch, time])
        """
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled, weights.squeeze(-1)


class FeatureAugmentation:
    """Feature-level augmentation methods."""
    
    def __init__(self, config):
        self.config = config
        self.time_mask_prob = config.get('time_mask_prob', 0.5)
        self.time_mask_ratio = config.get('time_mask_ratio', 0.1)
    
    def time_mask(self, embeddings):
        """Apply time masking to embeddings. Vectorized implementation."""
        if torch.rand(1).item() > self.time_mask_prob:
            return embeddings
        
        batch_size, seq_len, dim = embeddings.shape
        mask_len = int(seq_len * self.time_mask_ratio)
        if mask_len == 0:
            return embeddings

        start_indices = torch.randint(0, max(1, seq_len - mask_len), (batch_size,), device=embeddings.device)
        positions = torch.arange(seq_len, device=embeddings.device).expand(batch_size, seq_len)
        mask = (positions >= start_indices.unsqueeze(1)) & (positions < (start_indices + mask_len).unsqueeze(1))
        
        masked_embeddings = embeddings.clone().masked_fill_(mask.unsqueeze(-1), 0.0)
        return masked_embeddings


# ===================================================================
# =============        MAIN DISCRETE SSL MODEL        =============
# ===================================================================

class DiscreteSSLModel(nn.Module):
    """Model using discrete SSL tokens from K-means quantization."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
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
        
        ssl_model_type = config['ssl_model_type']
        if ssl_model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown SSL model type: {ssl_model_type}")
        
        cfg = MODEL_CONFIGS[ssl_model_type]
        
        cache_dir = config.get('cache_dir', 'pretrained_models')
        save_path = os.path.join(cache_dir, ssl_model_type)
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Create SSL model
        print(f"Loading {ssl_model_type} model from {cfg['source']}...")
        self.ssl_model = cfg["model_class"](
            source=cfg["source"],
            save_path=save_path,
            output_all_hiddens=True
        )
        
        # 2. Create DiscreteSSL wrapper
        print("Initializing DiscreteSSL wrapper...")
        self.discrete_ssl = DiscreteSSL(
            save_path=save_path,
            ssl_model=self.ssl_model,
            kmeans_dataset=config.get('kmeans_dataset', 'LibriSpeech960'),
            vocoder_repo_id=cfg["vocoder"],
            num_clusters=config.get('num_clusters', 1000),
            layers_num=None
        )
        
        # Freeze SSL model parameters
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        
        # Configure layers to use
        self.use_all_layers = config.get('use_all_layers', False)
        if hasattr(self.discrete_ssl, 'ssl_layer_ids'):
            available_layers = self.discrete_ssl.ssl_layer_ids
            print(f"Available SSL layers: {available_layers}")
        else:
            available_layers = [1, 3, 7, 12, 18, 23]
            print(f"Using default supported layers: {available_layers}")

        if self.use_all_layers:
            self.ssl_layers = available_layers
        else:
            self.ssl_layers = config.get('ssl_layers', [7])
        
        print(f"Using SSL layers: {self.ssl_layers}")
        self.num_layers = len(self.ssl_layers)
        self.num_clusters = config.get('num_clusters', 1000)
        self.ssl_downsample_rate = config.get('ssl_downsample_rate', 320)
        
        # Discrete embedding layer
        embedding_dim = config.get('embedding_dim', 256)
        self.embedding_layer = DiscreteEmbeddingLayer(
            num_clusters=self.num_clusters,
            embedding_dim=embedding_dim,
            ssl_layers=self.ssl_layers
        )
        
        # Layer aggregation if using multiple layers
        if self.use_all_layers and self.num_layers > 1:
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
            self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Backend: ECAPA-TDNN or Attention Pooling
        self.use_attention_pool = config.get('use_attention_pool', False)
        if self.use_attention_pool:
            self.backend = AttentionPooling(embedding_dim)
            classifier_input = embedding_dim
        else:
            self.backend = ECAPA_TDNN(
                input_dim=embedding_dim,
                channels=config.get('ecapa_channels', [512, 512, 512, 512, 1536]),
                lin_neurons=config.get('ecapa_lin_neurons', 192)
            )
            classifier_input = config.get('ecapa_lin_neurons', 192)
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_input),
            nn.ReLU(),
            nn.Dropout(config.get('classifier_dropout', 0.3)),
            nn.Linear(classifier_input, config['num_classes'])
        )
        
        # Feature augmentation
        if config.get('use_augmentation', False):
            self.feature_aug = FeatureAugmentation(config.get('augmentation', {}))
        else:
            self.feature_aug = None
        
        # Print model configuration
        print(f"\nModel initialized with:")
        print(f"  - Model type: {ssl_model_type}")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - Num clusters: {self.num_clusters}")
        print(f"  - Layers used: {self.ssl_layers}")
        print(f"  - Backend: {'Attention Pooling' if self.use_attention_pool else 'ECAPA-TDNN'}")
        print(f"  - Num classes: {config['num_classes']}")
    
    def extract_discrete_tokens(self, waveforms):
        """Extract discrete tokens using pre-trained SSL quantizer."""
        with torch.no_grad():
            tokens, _, _ = self.discrete_ssl.encode(
                waveforms,
                SSL_layers=self.ssl_layers,
                deduplicates=[False] * self.num_layers,
                bpe_tokenizers=[None] * self.num_layers
            )
        return tokens
    
    def forward(self, waveforms, lengths=None, training=False):
        """
        Args:
            waveforms (torch.Tensor): [batch, samples]
            lengths (torch.Tensor, optional): [batch] length ratios (0-1) relative to max_length
            training (bool, optional): Whether in training mode. Defaults to False.
        Returns:
            tuple: (logits [batch, num_classes], final_embeddings [batch, embedding_dim])
        """
        # Input validation
        if waveforms.dim() != 2:
            raise ValueError(f"Expected 2D waveforms [batch, samples], got {waveforms.dim()}D")
        
        batch_size = waveforms.size(0)
        device = waveforms.device
        
        # 1. Extract discrete tokens
        tokens = self.extract_discrete_tokens(waveforms)  # [batch, time, num_layers]
        token_seq_len = tokens.size(1)
        
        # 2. Convert tokens to embeddings
        embeddings = self.embedding_layer(tokens)  # [batch, time, num_layers, embedding_dim]
        
        # 3. Aggregate layers
        if self.use_all_layers and self.num_layers > 1:
            weights = F.softmax(self.layer_weights, dim=0).view(1, 1, -1, 1)
            embeddings = torch.sum(embeddings * weights, dim=2)
            embeddings = self.layer_norm(embeddings)
        else:
            # If using single layer, take the first (and possibly only) layer
            embeddings = embeddings[:, :, 0, :]
        
        # 4. Apply feature-level augmentation
        if training and self.feature_aug is not None:
            embeddings = self.feature_aug.time_mask(embeddings)
        
        # 5. Convert length ratios to token lengths
        token_lengths = None
        if lengths is not None:
            if lengths.dim() != 1:
                raise ValueError(f"Expected 1D lengths [batch], got {lengths.dim()}D")
            if (lengths < 0).any() or (lengths > 1).any():
                raise ValueError("Length ratios must be in range [0, 1]")
            
            if lengths.device != device:
                lengths = lengths.to(device)
            
            # Convert ratios to audio samples, then to token lengths
            audio_samples = (lengths * waveforms.size(1)).long()
            token_lengths = (audio_samples / self.ssl_downsample_rate).ceil().long()
            token_lengths = torch.clamp(token_lengths, min=1, max=token_seq_len)
        
        # 6. Process through backend
        if self.use_attention_pool:
            mask = None
            if token_lengths is not None:
                mask = torch.arange(token_seq_len, device=device).expand(batch_size, token_seq_len)
                mask = mask < token_lengths.unsqueeze(1)
            final_embeddings, _ = self.backend(embeddings, mask)
        else:
            # ECAPA-TDNN expects [batch, channels, time]
            embeddings_transposed = embeddings.transpose(1, 2)
            final_embeddings = self.backend(embeddings_transposed, token_lengths)
        
        # 7. Classification
        logits = self.classifier(final_embeddings)
        
        return logits, final_embeddings