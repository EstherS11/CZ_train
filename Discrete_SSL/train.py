import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import yaml
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import json

from model import DiscreteSSLModel
from dataset import get_dataloaders, MSPDataset
from utils import (
    get_loss_function, compute_metrics, get_classification_report,
    EarlyStopping, WarmupCosineScheduler, setup_logging, setup_wandb,
    save_checkpoint, load_checkpoint, count_parameters
)


class MixupAugmentation:
    """Mixup augmentation for embeddings - batch-level operation"""
    
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, embeddings, labels):
        """Apply mixup to embeddings and labels
        Args:
            embeddings: [batch, dim]
            labels: [batch]
        Returns:
            mixed_embeddings, mixed_labels (labels_a, labels_b, lam)
        """
        if np.random.random() > self.prob:
            return embeddings, labels
        
        batch_size = embeddings.size(0)
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(embeddings.device)
        
        # Mix embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
        
        # Return both sets of labels for mixed loss
        return mixed_embeddings, (labels, labels[index], lam)


def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


class DistributedTrainer:
    """Distributed trainer class for SER models"""
    
    def __init__(self, config, model_class, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Setup output directory (only on rank 0)
        self.output_dir = config['output_dir']
        if rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Wait for rank 0 to create directory
        if world_size > 1:
            dist.barrier()
        
        # Setup logging (only on rank 0)
        if rank == 0:
            self.logger = setup_logging(self.output_dir)
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"Using {world_size} GPUs for training")
        else:
            self.logger = None
        
        # Setup wandb (only on rank 0)
        self.use_wandb = False
        if rank == 0:
            self.use_wandb = setup_wandb(config)
        
        # Save config (only on rank 0)
        if rank == 0:
            with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
                yaml.dump(config, f)
        
        # Load data with distributed samplers
        self._setup_data_loaders()
        
        # Initialize model
        if rank == 0:
            self.logger.info(f"Initializing {model_class.__name__}...")
        
        self.model = model_class(config).to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)
        
        # Count parameters (only on rank 0)
        if rank == 0:
            trainable, total = count_parameters(self.model.module)
            self.logger.info(f"Trainable parameters: {trainable:,} / {total:,}")
        
        # Loss function
        self.criterion = get_loss_function(config, self.class_weights).to(self.device)
        
        # Optimizer (scale learning rate by world size)
        scaled_lr = config['lr'] * world_size
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=scaled_lr,
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Early stopping (only on rank 0)
        if rank == 0:
            self.early_stopping = EarlyStopping(
                patience=config.get('early_stopping_patience', 10),
                mode='max'
            )
        
        # Training state
        self.start_epoch = 0
        self.best_score = 0
        
        # Mixup augmentation (batch-level operation)
        if config.get('use_mixup', False):
            self.mixup = MixupAugmentation(
                alpha=config.get('mixup_alpha', 1.0),
                prob=config.get('mixup_prob', 0.5)
            )
        else:
            self.mixup = None
    
    def _setup_data_loaders(self):
        """Create distributed data loaders"""
        # Create datasets
        train_dataset = MSPDataset(
            self.config['train_json'],
            self.config['root_dir'],
            self.config['max_length'],
            fixed_length=self.config.get('use_fixed_length', True)
        )
        
        valid_dataset = MSPDataset(
            self.config['valid_json'],
            self.config['root_dir'],
            self.config['max_length'],
            fixed_length=self.config.get('use_fixed_length', True)
        )
        
        test_dataset = MSPDataset(
            self.config['test_json'],
            self.config['root_dir'],
            self.config['max_length'],
            fixed_length=self.config.get('use_fixed_length', True)
        )
        
        # Get class weights
        self.class_weights = train_dataset.class_weights
        
        # Print class distribution (only on rank 0)
        if self.rank == 0:
            print("\nClass distribution in training set:")
            for i, (name, count) in enumerate(zip(MSPDataset.EMOTION_NAMES, train_dataset.class_counts)):
                print(f"{name:10s}: {count:5d} samples, weight: {train_dataset.class_weights[i]:.3f}")
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=42
        )
        
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Choose appropriate collate function
        from dataset import collate_fn_dynamic, collate_fn_fixed
        collate_fn = collate_fn_dynamic if not self.config.get('use_fixed_length', True) else collate_fn_fixed
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'] // self.world_size,  # Divide batch size by world size
            sampler=train_sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )
        
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.config['batch_size'] // self.world_size,
            sampler=valid_sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'] // self.world_size,
            sampler=test_sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )
        
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.test_sampler = test_sampler
    
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        sched_type = self.config.get('scheduler_type', 'cosine')
        
        if sched_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif sched_type == 'warmup_cosine':
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.get('warmup_epochs', 5),
                total_epochs=self.config['num_epochs'],
                min_lr=self.config.get('min_lr', 1e-6)
            )
        return None
    
    def compute_loss(self, logits, mixed_labels):
        """Compute loss with mixup support"""
        if isinstance(mixed_labels, tuple):
            # Mixup loss
            labels_a, labels_b, lam = mixed_labels
            loss_a = self.criterion(logits, labels_a)
            loss_b = self.criterion(logits, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            # Standard loss
            loss = self.criterion(logits, mixed_labels)
        return loss
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Create progress bar only on rank 0
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            waveforms = batch['waveforms'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch.get('lengths', None)
            if lengths is not None:
                lengths = lengths.to(self.device)
            
            # Forward pass (model handles time masking internally)
            logits, embeddings = self.model(waveforms, lengths, training=True)
            
            # Apply mixup if enabled (batch-level operation)
            mixed_labels = labels
            if self.mixup is not None:
                mixed_embeddings, mixed_labels = self.mixup(embeddings, labels)
                if isinstance(mixed_labels, tuple):
                    # Re-classify mixed embeddings
                    logits = self.model.module.classifier(mixed_embeddings)
            
            # Calculate loss
            loss = self.compute_loss(logits, mixed_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Only collect predictions if no mixup
            if not isinstance(mixed_labels, tuple):
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar (only on rank 0)
            if self.rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Log to wandb (only on rank 0)
            if self.rank == 0 and self.use_wandb and batch_idx % 10 == 0:
                import wandb
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        
        # Gather predictions from all processes
        if len(all_preds) > 0:
            # Convert to tensors
            preds_tensor = torch.tensor(all_preds, device=self.device)
            labels_tensor = torch.tensor(all_labels, device=self.device)
            
            # Gather all predictions
            world_preds = [torch.zeros_like(preds_tensor) for _ in range(self.world_size)]
            world_labels = [torch.zeros_like(labels_tensor) for _ in range(self.world_size)]
            
            dist.all_gather(world_preds, preds_tensor)
            dist.all_gather(world_labels, labels_tensor)
            
            if self.rank == 0:
                all_preds = torch.cat(world_preds).cpu().numpy()
                all_labels = torch.cat(world_labels).cpu().numpy()
                metrics = compute_metrics(all_preds, all_labels)
            else:
                metrics = {'macro_f1': 0, 'accuracy': 0}
        else:
            metrics = {'macro_f1': 0, 'accuracy': 0}
        
        # Reduce average loss across all processes
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = reduce_tensor(avg_loss_tensor, self.world_size).item()
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def evaluate(self, loader, sampler, split='valid'):
        """Evaluate on given loader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_utterance_ids = []
        
        # Create progress bar only on rank 0
        if self.rank == 0:
            pbar = tqdm(loader, desc=f'Evaluating {split}')
        else:
            pbar = loader
        
        for batch in pbar:
            waveforms = batch['waveforms'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch.get('lengths', None)
            if lengths is not None:
                lengths = lengths.to(self.device)
            
            # Forward pass (no augmentation during evaluation)
            logits, _ = self.model(waveforms, lengths, training=False)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_utterance_ids.extend(batch['utterance_ids'])
        
        # Calculate metrics
        avg_loss = total_loss / len(loader)
        
        # Gather predictions from all processes
        preds_tensor = torch.tensor(all_preds, device=self.device)
        labels_tensor = torch.tensor(all_labels, device=self.device)
        
        # Gather all predictions
        world_preds = [torch.zeros_like(preds_tensor) for _ in range(self.world_size)]
        world_labels = [torch.zeros_like(labels_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(world_preds, preds_tensor)
        dist.all_gather(world_labels, labels_tensor)
        
        # Reduce average loss
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = reduce_tensor(avg_loss_tensor, self.world_size).item()
        
        if self.rank == 0:
            all_preds = torch.cat(world_preds).cpu().numpy()
            all_labels = torch.cat(world_labels).cpu().numpy()
            metrics = compute_metrics(all_preds, all_labels)
            
            # Get detailed report for test set
            if split == 'test':
                report = get_classification_report(
                    all_preds, all_labels, MSPDataset.EMOTION_NAMES
                )
                # Note: utterance_ids would need to be gathered as well for complete results
                self.save_test_results([], all_preds, all_labels, metrics, report)
        else:
            metrics = {'macro_f1': 0, 'accuracy': 0}
        
        return avg_loss, metrics
    
    def save_test_results(self, utterance_ids, predictions, labels, metrics, report):
        """Save detailed test results"""
        results = {
            'metrics': {
                'macro_f1': float(metrics['macro_f1']),
                'accuracy': float(metrics['accuracy']),
                'per_class_f1': {name: float(f1) for name, f1 in 
                               zip(MSPDataset.EMOTION_NAMES, metrics['per_class_f1'])}
            },
            'classification_report': report,
            'predictions': []
        }
        
        # Note: In distributed setting, we might not have all utterance_ids
        # This is simplified for the metrics only
        
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    def train(self):
        """Main training loop"""
        if self.rank == 0:
            self.logger.info("Starting distributed training...")
        
        # Resume if checkpoint exists (only load on rank 0 first)
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path) and self.config.get('resume', True):
            if self.rank == 0:
                self.start_epoch, self.best_score = load_checkpoint(
                    checkpoint_path, self.model.module, self.optimizer, self.scheduler
                )
                self.logger.info(f"Resumed from epoch {self.start_epoch}")
            
            # Broadcast epoch and best score to all processes
            epoch_tensor = torch.tensor(self.start_epoch, device=self.device)
            score_tensor = torch.tensor(self.best_score, device=self.device)
            dist.broadcast(epoch_tensor, 0)
            dist.broadcast(score_tensor, 0)
            
            if self.rank != 0:
                self.start_epoch = epoch_tensor.item()
                self.best_score = score_tensor.item()
        
        # Training loop
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            valid_loss, valid_metrics = self.evaluate(self.valid_loader, self.valid_sampler, 'valid')
            
            # Update scheduler
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
            
            # Log results (only on rank 0)
            if self.rank == 0:
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['macro_f1']:.4f}, "
                    f"Valid Loss: {valid_loss:.4f}, Valid F1: {valid_metrics['macro_f1']:.4f}"
                )
                
                # Log to wandb
                if self.use_wandb:
                    import wandb
                    log_dict = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_macro_f1': train_metrics['macro_f1'],
                        'train_accuracy': train_metrics['accuracy'],
                        'valid_loss': valid_loss,
                        'valid_macro_f1': valid_metrics['macro_f1'],
                        'valid_accuracy': valid_metrics['accuracy'],
                    }
                    
                    # Add per-class F1
                    for i, name in enumerate(MSPDataset.EMOTION_NAMES):
                        log_dict[f'valid_f1_{name}'] = valid_metrics['per_class_f1'][i]
                    
                    wandb.log(log_dict)
                
                # Save checkpoint
                is_best = valid_metrics['macro_f1'] > self.best_score
                if is_best:
                    self.best_score = valid_metrics['macro_f1']
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_score': self.best_score,
                    'config': self.config
                }, 'checkpoint.pth', is_best, self.output_dir)
                
                # Early stopping
                if self.early_stopping(valid_metrics['macro_f1']):
                    self.logger.info("Early stopping triggered!")
                    break
            
            # Synchronize all processes
            dist.barrier()
            
            # Check if early stopping was triggered on rank 0
            if self.rank == 0 and self.early_stopping.early_stop:
                stop_tensor = torch.tensor(1, device=self.device)
            else:
                stop_tensor = torch.tensor(0, device=self.device)
            
            dist.broadcast(stop_tensor, 0)
            if stop_tensor.item() == 1:
                break
        
        # Test on best model (only on rank 0)
        if self.rank == 0:
            self.logger.info("Loading best model for testing...")
            load_checkpoint(
                os.path.join(self.output_dir, 'best_model.pth'),
                self.model.module
            )
        
        # Synchronize before testing
        dist.barrier()
        
        test_loss, test_metrics = self.evaluate(self.test_loader, self.test_sampler, 'test')
        
        if self.rank == 0:
            self.logger.info(f"\nTest Results:")
            self.logger.info(f"Loss: {test_loss:.4f}")
            self.logger.info(f"Macro F1: {test_metrics['macro_f1']:.4f}")
            self.logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
            
            self.logger.info("\nPer-class F1 scores:")
            for i, name in enumerate(MSPDataset.EMOTION_NAMES):
                self.logger.info(f"{name:10s}: {test_metrics['per_class_f1'][i]:.4f}")
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    'test_loss': test_loss,
                    'test_macro_f1': test_metrics['macro_f1'],
                    'test_accuracy': test_metrics['accuracy']
                })
                wandb.finish()


def run_distributed(rank, world_size, config, model_class):
    """Run distributed training on a single process"""
    # Setup distributed environment
    setup_distributed(rank, world_size)
    
    try:
        # Create trainer and run training
        trainer = DistributedTrainer(config, model_class, rank, world_size)
        trainer.train()
    finally:
        # Clean up
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--ssl_model', type=str, required=True,
                       choices=['wav2vec2', 'hubert', 'wavlm'],
                       help='SSL model type')
    parser.add_argument('--gpus', type=int, default=4,
                       help='Number of GPUs to use (default: 4)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for discrete model
    config['ssl_model_type'] = args.ssl_model
    
    # Update experiment name and output dir
    config['experiment_name'] = f"msp_{args.ssl_model}_discrete_distributed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config['output_dir'] = f"experiments/{config['experiment_name']}"
    
    # Set number of GPUs
    world_size = args.gpus
    
    # Check if running with torchrun or as standalone script
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        run_distributed(rank, world_size, config, DiscreteSSLModel)
    else:
        # Running as standalone script - spawn processes
        import torch.multiprocessing as mp
        mp.spawn(
            run_distributed,
            args=(world_size, config, DiscreteSSLModel),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    main()