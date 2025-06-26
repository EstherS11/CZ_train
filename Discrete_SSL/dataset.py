import torch
import torchaudio
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import numpy as np


class MSPDataset(Dataset):
    """MSP-PODCAST Dataset with dynamic length support"""
    
    EMOTION_MAP = {
        'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
        'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
    }
    
    EMOTION_NAMES = [
        "neutral", "happy", "sad", "angry", "fear",
        "disgust", "surprise", "contempt", "other", "unknown"
    ]
    
    def __init__(self, json_path, root_dir, max_length=10.0, fixed_length=False):
        """
        Args:
            json_path: Path to json file with annotations
            root_dir: Root directory containing audio files
            max_length: Maximum audio length in seconds (for length ratio calculation)
            fixed_length: If True, pad/truncate to max_length; if False, return variable lengths
        """
        self.root_dir = Path(root_dir)
        self.max_length = max_length
        self.sample_rate = 16000
        self.fixed_length = fixed_length
        
        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.utterance_ids = list(self.data.keys())
        
        # Calculate class weights
        self.class_counts = [0] * 10
        for utt_id in self.utterance_ids:
            label = self.EMOTION_MAP[self.data[utt_id]['emo']]
            self.class_counts[label] += 1
        
        total = len(self.utterance_ids)
        self.class_weights = [total / (10 * count) if count > 0 else 0 
                             for count in self.class_counts]
    
    def get_sample_weights(self):
        """Get sample weights efficiently without loading audio files"""
        weights = []
        for utt_id in self.utterance_ids:
            emo = self.data[utt_id]['emo']
            label = self.EMOTION_MAP[emo]
            weights.append(self.class_weights[label])
        return weights
    
    def __len__(self):
        return len(self.utterance_ids)
    
    def __getitem__(self, idx):
        utt_id = self.utterance_ids[idx]
        item = self.data[utt_id]
        
        # Load audio
        wav_path = self.root_dir / item['wav']
        waveform, sr = torchaudio.load(wav_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Get label
        label = self.EMOTION_MAP[item['emo']]
        
        # Handle fixed vs variable length
        if self.fixed_length:
            # Fixed length: pad or truncate
            max_samples = int(self.max_length * self.sample_rate)
            if waveform.shape[1] > max_samples:
                # Center crop for fixed length
                start = (waveform.shape[1] - max_samples) // 2
                waveform = waveform[:, start:start + max_samples]
            else:
                # Pad with zeros
                pad_length = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # For fixed length, actual length equals max length
            actual_length = self.max_length
        else:
            # Variable length: keep original length
            actual_length = min(waveform.shape[1] / self.sample_rate, self.max_length)
        
        return {
            'utterance_id': utt_id,
            'waveform': waveform.squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(actual_length / self.max_length, dtype=torch.float),  # This is a ratio!
            'actual_samples': waveform.shape[1]  # Actual number of samples
        }


def collate_fn_dynamic(batch):
    """Custom collate function for variable length sequences"""
    # Find the maximum length in this batch
    max_samples = max(item['actual_samples'] for item in batch)
    
    # Prepare tensors
    batch_size = len(batch)
    waveforms = torch.zeros(batch_size, max_samples)
    labels = torch.zeros(batch_size, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.float)
    utterance_ids = []
    
    # Fill the tensors
    for i, item in enumerate(batch):
        wf = item['waveform']
        actual_samples = item['actual_samples']
        
        # Pad waveform to max length in batch
        waveforms[i, :actual_samples] = wf
        labels[i] = item['label']
        lengths[i] = item['length']
        utterance_ids.append(item['utterance_id'])
    
    return {
        'utterance_ids': utterance_ids,
        'waveforms': waveforms,
        'labels': labels,
        'lengths': lengths  # These are ratios (0-1)
    }


def collate_fn_fixed(batch):
    """Simple collate function for fixed length sequences"""
    utterance_ids = [item['utterance_id'] for item in batch]
    waveforms = torch.stack([item['waveform'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    
    return {
        'utterance_ids': utterance_ids,
        'waveforms': waveforms,
        'labels': labels,
        'lengths': lengths
    }


def get_dataloaders(config):
    """Create data loaders with efficient sampling"""
    
    # Determine if using fixed or variable length
    use_fixed_length = config.get('use_fixed_length', True)
    
    # Create datasets
    train_dataset = MSPDataset(
        config['train_json'],
        config['root_dir'],
        config['max_length'],
        fixed_length=use_fixed_length
    )
    
    valid_dataset = MSPDataset(
        config['valid_json'],
        config['root_dir'],
        config['max_length'],
        fixed_length=use_fixed_length
    )
    
    test_dataset = MSPDataset(
        config['test_json'],
        config['root_dir'],
        config['max_length'],
        fixed_length=use_fixed_length
    )
    
    # Print class distribution
    print("\nClass distribution in training set:")
    for i, (name, count) in enumerate(zip(MSPDataset.EMOTION_NAMES, train_dataset.class_counts)):
        print(f"{name:10s}: {count:5d} samples, weight: {train_dataset.class_weights[i]:.3f}")
    
    # Efficient weighted sampler (no audio loading!)
    sampler = None
    if config.get('use_balanced_sampling', True):
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Choose appropriate collate function
    collate_fn = collate_fn_dynamic if not use_fixed_length else collate_fn_fixed
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    # Return as dictionary for clarity
    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
    
    return dataloaders, train_dataset.class_weights