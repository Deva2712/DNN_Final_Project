#!/usr/bin/env python3
"""
Training Script

Trains disagreement prediction models with different loss functions.
Supports both pretraining and fine-tuning phases with configurable hyperparameters.

Usage:
    python train.py [options]

Examples:
    # Train all three models (KL, JS, Custom) with default settings
    python train.py

    # Train only KL model
    python train.py --loss-functions kl

    # Custom hyperparameters
    python train.py --pretrain-epochs 50 --finetune-epochs 30 --finetune-lr 5e-5

    # Skip pretraining (load existing pretrained model)
    python train.py --skip-pretrain --pretrained-path checkpoints/pretrained_resnet18_cifar10.pth

    # Train on CPU
    python train.py --device cpu
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DisagreementPredictor, ModelConfig
from src.data_pipeline import (
    load_cifar10_data,
    load_cifar10h_data,
    compute_soft_labels,
    align_datasets,
    split_dataset,
    compute_entropy,
    CIFAR10HDataset
)
from src.training import (
    set_seed,
    get_train_transform,
    get_test_transform,
    pretrain_on_hard_labels,
    finetune_on_soft_labels,
    TrainingConfig
)
from src.losses import kl_divergence_loss, js_divergence_loss, custom_entropy_regularized_loss
from src.logging_config import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CIFAR-10 disagreement prediction models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data directories
    parser.add_argument(
        '--cifar10-dir',
        type=str,
        default='./data',
        help='Directory containing CIFAR-10 dataset'
    )
    parser.add_argument(
        '--cifar10h-dir',
        type=str,
        default='./cifar-10h-1.0.0/data',
        help='Directory containing CIFAR-10H dataset'
    )
    
    # Output directories
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./outputs/training_logs',
        help='Directory to save training logs'
    )
    
    # Loss functions
    parser.add_argument(
        '--loss-functions',
        type=str,
        nargs='+',
        default=['kl', 'js', 'custom'],
        choices=['kl', 'js', 'custom'],
        help='Loss functions to train with'
    )
    
    # Pretraining hyperparameters
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=100,
        help='Number of pretraining epochs'
    )
    parser.add_argument(
        '--pretrain-lr',
        type=float,
        default=1e-3,
        help='Learning rate for pretraining'
    )
    parser.add_argument(
        '--pretrain-batch-size',
        type=int,
        default=128,
        help='Batch size for pretraining'
    )
    
    # Fine-tuning hyperparameters
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=50,
        help='Maximum number of fine-tuning epochs'
    )
    parser.add_argument(
        '--finetune-lr',
        type=float,
        default=1e-4,
        help='Learning rate for fine-tuning'
    )
    parser.add_argument(
        '--finetune-batch-size',
        type=int,
        default=64,
        help='Batch size for fine-tuning'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for AdamW optimizer'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Patience for early stopping'
    )
    
    # Custom loss hyperparameter
    parser.add_argument(
        '--lambda-weight',
        type=float,
        default=0.1,
        help='Weight for entropy penalty in custom loss'
    )
    
    # Pretraining options
    parser.add_argument(
        '--skip-pretrain',
        action='store_true',
        help='Skip pretraining phase (load existing pretrained model)'
    )
    parser.add_argument(
        '--pretrained-path',
        type=str,
        default='checkpoints/pretrained_resnet18_cifar10.pth',
        help='Path to pretrained model weights'
    )
    
    # Device and seed
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    return parser.parse_args()


def prepare_data(args, logger):
    """Prepare datasets and dataloaders."""
    logger.info("="*70)
    logger.info("Preparing datasets")
    logger.info("="*70)
    
    # Load CIFAR-10 test set for CIFAR-10H
    cifar10_images, cifar10_labels = load_cifar10_data(
        data_dir=args.cifar10_dir,
        train=False,
        download=True
    )
    logger.info(f"✓ Loaded CIFAR-10 test set: {len(cifar10_images)} images")
    
    # Load CIFAR-10H
    cifar10h_counts, cifar10h_probs = load_cifar10h_data(data_dir=args.cifar10h_dir)
    soft_labels = compute_soft_labels(cifar10h_counts)
    logger.info(f"✓ Loaded CIFAR-10H: {len(soft_labels)} soft labels")
    
    # Align and split
    aligned_data = align_datasets(cifar10_images, cifar10_labels, soft_labels)
    train_data, val_data, test_data = split_dataset(aligned_data, random_seed=args.seed)
    logger.info(f"✓ Split dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Compute entropies
    entropies = compute_entropy(soft_labels)
    
    # Create PyTorch datasets
    def create_dataset(data_list, transform=None):
        images = torch.stack([torch.from_numpy(img).float() / 255.0 for img, _, _ in data_list])
        soft_labels_tensor = torch.stack([torch.from_numpy(sl).float() for _, sl, _ in data_list])
        hard_labels_tensor = torch.tensor([hl for _, _, hl in data_list], dtype=torch.long)
        
        # Get entropies for these samples
        indices = []
        for img, _, _ in data_list:
            idx = np.where((cifar10_images == img).all(axis=(1, 2, 3)))[0][0]
            indices.append(idx)
        entropies_tensor = torch.from_numpy(entropies[indices]).float()
        
        return CIFAR10HDataset(images, soft_labels_tensor, hard_labels_tensor, entropies_tensor, transform)
    
    import numpy as np
    train_dataset = create_dataset(train_data, transform=get_train_transform())
    val_dataset = create_dataset(val_data, transform=get_test_transform())
    
    # Create dataloaders for fine-tuning
    train_loader_finetune = DataLoader(
        train_dataset,
        batch_size=args.finetune_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.finetune_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Create dataloader for pretraining (CIFAR-10 training set with hard labels)
    if not args.skip_pretrain:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616])
        ])
        
        cifar10_train = datasets.CIFAR10(
            root=args.cifar10_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        
        train_loader_pretrain = DataLoader(
            cifar10_train,
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if args.device == 'cuda' else False
        )
        logger.info(f"✓ Created pretraining dataloader: {len(cifar10_train)} samples")
    else:
        train_loader_pretrain = None
    
    logger.info(f"✓ Created fine-tuning dataloaders")
    
    return train_loader_pretrain, train_loader_finetune, val_loader


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("CIFAR-10 Human Disagreement Predictor - Training")
    logger.info("="*70)
    logger.info(f"Device: {args.device}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Loss functions: {', '.join(args.loss_functions)}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Prepare data
    train_loader_pretrain, train_loader_finetune, val_loader = prepare_data(args, logger)
    
    # Phase 1: Pretraining
    if not args.skip_pretrain:
        logger.info("\n" + "="*70)
        logger.info("Phase 1: Pretraining on CIFAR-10 hard labels")
        logger.info("="*70)
        
        model = DisagreementPredictor()
        
        try:
            model, pretrain_history = pretrain_on_hard_labels(
                model=model,
                train_loader=train_loader_pretrain,
                num_epochs=args.pretrain_epochs,
                device=args.device,
                save_path=args.pretrained_path
            )
            logger.info(f"✓ Pretraining complete")
            logger.info(f"  Final training accuracy: {pretrain_history['train_acc'][-1]:.2f}%")
            
            # Save pretraining history
            import json
            history_path = os.path.join(args.log_dir, 'pretrain_history.json')
            with open(history_path, 'w') as f:
                json.dump(pretrain_history, f, indent=2)
            logger.info(f"✓ Saved pretraining history to {history_path}")
            
        except Exception as e:
            logger.error(f"Pretraining failed: {e}")
            sys.exit(1)
    else:
        logger.info(f"\nSkipping pretraining, will load from {args.pretrained_path}")
    
    # Phase 2: Fine-tuning with different loss functions
    logger.info("\n" + "="*70)
    logger.info("Phase 2: Fine-tuning on CIFAR-10H soft labels")
    logger.info("="*70)
    
    loss_functions = {
        'kl': kl_divergence_loss,
        'js': js_divergence_loss,
        'custom': lambda pred, target: custom_entropy_regularized_loss(
            pred, target, lambda_weight=args.lambda_weight
        )
    }
    
    results = {}
    
    for loss_name in args.loss_functions:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training with {loss_name.upper()} loss")
        logger.info(f"{'='*70}")
        
        # Create new model and load pretrained weights
        model = DisagreementPredictor()
        
        if os.path.exists(args.pretrained_path):
            model.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))
            logger.info(f"✓ Loaded pretrained weights from {args.pretrained_path}")
        else:
            logger.warning(f"Pretrained weights not found at {args.pretrained_path}, using random initialization")
        
        # Fine-tune
        save_path = os.path.join(args.checkpoint_dir, f'finetuned_{loss_name}_best.pth')
        
        try:
            model, finetune_history = finetune_on_soft_labels(
                model=model,
                train_loader=train_loader_finetune,
                val_loader=val_loader,
                loss_fn=loss_functions[loss_name],
                loss_name=loss_name,
                num_epochs=args.finetune_epochs,
                device=args.device,
                save_path=save_path
            )
            
            logger.info(f"✓ Fine-tuning with {loss_name.upper()} loss complete")
            logger.info(f"  Best validation KL: {min(finetune_history['val_kl']):.4f}")
            logger.info(f"  Best validation JS: {min(finetune_history['val_js']):.4f}")
            
            results[loss_name] = {
                'model': model,
                'history': finetune_history,
                'checkpoint_path': save_path
            }
            
            # Save fine-tuning history
            import json
            history_path = os.path.join(args.log_dir, f'finetune_{loss_name}_history.json')
            with open(history_path, 'w') as f:
                json.dump(finetune_history, f, indent=2)
            logger.info(f"✓ Saved fine-tuning history to {history_path}")
            
        except Exception as e:
            logger.error(f"Fine-tuning with {loss_name} loss failed: {e}")
            continue
    
    # Save training configuration
    logger.info("\n" + "="*70)
    logger.info("Saving training configuration")
    logger.info("="*70)
    
    try:
        config = TrainingConfig(
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            pretrain_lr=args.pretrain_lr,
            finetune_lr=args.finetune_lr,
            weight_decay=args.weight_decay,
            pretrain_batch_size=args.pretrain_batch_size,
            finetune_batch_size=args.finetune_batch_size,
            early_stopping_patience=args.early_stopping_patience,
            random_seed=args.seed,
            lambda_weight=args.lambda_weight
        )
        config_path = os.path.join(args.log_dir, 'training_config.json')
        config.to_json(config_path)
        logger.info(f"✓ Saved training configuration to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save training configuration: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nTrained {len(results)} model(s):")
    for loss_name, result in results.items():
        logger.info(f"  {loss_name.upper()}: {result['checkpoint_path']}")
        logger.info(f"    Best Val KL: {min(result['history']['val_kl']):.4f}")
        logger.info(f"    Best Val JS: {min(result['history']['val_js']):.4f}")
    logger.info(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    logger.info(f"Training logs saved to: {args.log_dir}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
