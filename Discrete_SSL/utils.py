import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import wandb
import logging
import os
from datetime import datetime


# ============= Loss Functions =============

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=-1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p[torch.arange(targets.size(0)), targets]
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    def __init__(self, epsilon=0.1, reduction='mean', weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
    
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.epsilon) + self.epsilon / n_classes
        
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(smoothed * log_probs).sum(dim=-1)
        
        if self.weight is not None:
            loss = loss * self.weight[targets]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MixupCriterion:
    """Criterion for mixup training"""
    def __init__(self, base_criterion):
        self.base_criterion = base_criterion
    
    def __call__(self, outputs, targets, mixup_lambda=None, mixup_index=None):
        if mixup_lambda is None:
            return self.base_criterion(outputs, targets)
        
        loss1 = self.base_criterion(outputs, targets)
        loss2 = self.base_criterion(outputs, targets[mixup_index])
        return mixup_lambda * loss1 + (1 - mixup_lambda) * loss2


def get_loss_function(config, class_weights=None):
    """Get loss function based on config"""
    loss_type = config.get('loss_type', 'focal')
    
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        weights = None
    
    if loss_type == 'focal':
        base_loss = FocalLoss(alpha=weights, gamma=config.get('focal_gamma', 2.0))
    elif loss_type == 'ce':
        base_loss = nn.CrossEntropyLoss(weight=weights)
    elif loss_type == 'label_smoothing':
        base_loss = LabelSmoothingCrossEntropy(
            epsilon=config.get('label_smoothing_epsilon', 0.1),
            weight=weights
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Wrap with mixup if enabled
    if config.get('use_mixup', False):
        return MixupCriterion(base_loss)
    
    return base_loss


# ============= Metrics =============

def compute_metrics(predictions, labels, num_classes=10):
    """Compute evaluation metrics"""
    macro_f1 = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    per_class_f1 = f1_score(labels, predictions, average=None, labels=list(range(num_classes)))
    
    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'per_class_f1': per_class_f1
    }


def get_classification_report(predictions, labels, class_names):
    """Get detailed classification report"""
    return classification_report(
        labels, predictions, 
        target_names=class_names,
        digits=4,
        output_dict=True
    )


# ============= Training Utilities =============

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
            factor = factor * (1 - self.min_lr) + self.min_lr
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor
        
        return self.optimizer.param_groups[0]['lr']


# ============= Logging =============

def setup_logging(output_dir, log_level=logging.INFO):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def setup_wandb(config):
    """Initialize Weights & Biases"""
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'msp-podcast-ser'),
            name=config.get('experiment_name', None),
            config=config,
            tags=config.get('wandb_tags', [])
        )
        return True
    return False


# ============= Checkpointing =============

def save_checkpoint(state, filename, is_best=False, output_dir='./'):
    """Save model checkpoint"""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_score', 0)


# ============= Other Utilities =============

def count_parameters(model):
    """Count trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_optimizer_groups(model, config):
    """Get parameter groups with different learning rates"""
    ssl_params = []
    ecapa_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'ssl_model' in name:
            ssl_params.append(param)
        elif 'ecapa' in name:
            ecapa_params.append(param)
        else:
            other_params.append(param)
    
    groups = [
        {'params': other_params, 'lr': config['lr']},
        {'params': ecapa_params, 'lr': config.get('ecapa_lr', config['lr'])},
        {'params': ssl_params, 'lr': config.get('ssl_lr', config['lr'] * 0.1)}
    ]
    
    return [g for g in groups if len(g['params']) > 0]