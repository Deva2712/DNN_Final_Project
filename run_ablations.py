#!/usr/bin/env python3
"""
Ablation Study Script

Runs comprehensive ablation experiments to understand the impact of different
design choices: loss functions, initialization strategies, training strategies,
and prediction head architectures.

Usage:
    python run_ablations.py [options]

Examples:
    # Run all ablation studies
    python run_ablations.py

    # Run specific ablation studies
    python run_ablations.py --studies loss initialization

    # Custom epochs for faster experimentation
    python run_ablations.py --pretrain-epochs 20 --finetune-epochs 10
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DisagreementPredictor, DisagreementPredictionHead
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
    finetune_on_soft_labels
)
from src.losses import kl_divergence_loss, js_divergence_loss, custom_entropy_regularized_loss
from src.evaluation import (
    evaluate_model,
    compare_loss_functions,
    compare_backbone_initialization,
    compare_training_strategies,
    compare_prediction_head_architectures
)
from src.logging_config import setup_logging
import torch.nn as nn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run ablation studies for CIFAR-10 disagreement prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Ablation studies to run
    parser.add_argument(
        '--studies',
        type=str,
        nargs='+',
        default=['loss', 'initialization', 'training_strategy', 'architecture'],
        choices=['loss', 'initialization', 'training_strategy', 'architecture', 'all'],
        help='Which ablation studies to run'
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
    
    # Output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/ablation_studies',
        help='Directory to save ablation results'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/ablations',
        help='Directory to save ablation model checkpoints'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=100,
        help='Number of pretraining epochs'
    )
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=50,
        help='Number of fine-tuning epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training and evaluation'
    )
    
    # Device and seed
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
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
    logger.info("Preparing datasets...")
    
    # Load data
    cifar10_images, cifar10_labels = load_cifar10_data(args.cifar10_dir, train=False, download=True)
    cifar10h_counts, _ = load_cifar10h_data(args.cifar10h_dir)
    soft_labels = compute_soft_labels(cifar10h_counts)
    
    # Align and split
    aligned_data = align_datasets(cifar10_images, cifar10_labels, soft_labels)
    train_data, val_data, test_data = split_dataset(aligned_data, random_seed=args.seed)
    
    # Compute entropies
    entropies = compute_entropy(soft_labels)
    
    # Create datasets
    import numpy as np
    
    def create_dataset(data_list, transform=None):
        images = torch.stack([torch.from_numpy(img).float() / 255.0 for img, _, _ in data_list])
        soft_labels_tensor = torch.stack([torch.from_numpy(sl).float() for _, sl, _ in data_list])
        hard_labels_tensor = torch.tensor([hl for _, _, hl in data_list], dtype=torch.long)
        
        indices = []
        for img, _, _ in data_list:
            idx = np.where((cifar10_images == img).all(axis=(1, 2, 3)))[0][0]
            indices.append(idx)
        entropies_tensor = torch.from_numpy(entropies[indices]).float()
        
        return CIFAR10HDataset(images, soft_labels_tensor, hard_labels_tensor, entropies_tensor, transform)
    
    train_dataset = create_dataset(train_data, transform=get_train_transform())
    val_dataset = create_dataset(val_data, transform=get_test_transform())
    test_dataset = create_dataset(test_data, transform=get_test_transform())
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Pretraining dataloader
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    cifar10_train = datasets.CIFAR10(root=args.cifar10_dir, train=True, download=True, transform=train_transform)
    pretrain_loader = DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=4)
    
    logger.info(f"✓ Datasets prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return pretrain_loader, train_loader, val_loader, test_loader


def run_loss_function_ablation(pretrain_loader, train_loader, val_loader, test_loader, args, logger):
    """Ablation study: Compare different loss functions."""
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY: Loss Functions")
    logger.info("="*70)
    
    loss_functions = {
        'kl': kl_divergence_loss,
        'js': js_divergence_loss,
        'custom': custom_entropy_regularized_loss
    }
    
    models = {}
    
    # Pretrain once
    logger.info("\nPretraining backbone...")
    pretrained_path = os.path.join(args.checkpoint_dir, 'pretrained_for_loss_ablation.pth')
    
    if not os.path.exists(pretrained_path):
        model = DisagreementPredictor()
        model, _ = pretrain_on_hard_labels(
            model, pretrain_loader, num_epochs=args.pretrain_epochs,
            device=args.device, save_path=pretrained_path
        )
        logger.info(f"✓ Pretrained model saved to {pretrained_path}")
    else:
        logger.info(f"✓ Using existing pretrained model from {pretrained_path}")
    
    # Fine-tune with each loss function
    for loss_name, loss_fn in loss_functions.items():
        logger.info(f"\nFine-tuning with {loss_name.upper()} loss...")
        
        model = DisagreementPredictor()
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        
        save_path = os.path.join(args.checkpoint_dir, f'loss_ablation_{loss_name}.pth')
        model, _ = finetune_on_soft_labels(
            model, train_loader, val_loader, loss_fn, loss_name,
            num_epochs=args.finetune_epochs, device=args.device, save_path=save_path
        )
        
        models[loss_name] = model
        logger.info(f"✓ {loss_name.upper()} model trained")
    
    # Compare models
    logger.info("\nComparing loss functions...")
    comparison_df = compare_loss_functions(
        models, test_loader, device=args.device,
        output_dir=os.path.join(args.output_dir, 'loss_functions')
    )
    
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info("\n✓ Loss function ablation complete")
    
    return comparison_df


def run_initialization_ablation(pretrain_loader, train_loader, val_loader, test_loader, args, logger):
    """Ablation study: Compare initialization strategies."""
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY: Backbone Initialization")
    logger.info("="*70)
    
    models = {}
    
    # Strategy 1: Random initialization (no pretraining)
    logger.info("\nTraining with random initialization...")
    model_random = DisagreementPredictor()
    save_path = os.path.join(args.checkpoint_dir, 'init_ablation_random.pth')
    model_random, _ = finetune_on_soft_labels(
        model_random, train_loader, val_loader, kl_divergence_loss, 'random',
        num_epochs=args.finetune_epochs, device=args.device, save_path=save_path
    )
    models['random'] = model_random
    logger.info("✓ Random initialization model trained")
    
    # Strategy 2: CIFAR-10 pretraining
    logger.info("\nTraining with CIFAR-10 pretraining...")
    pretrained_path = os.path.join(args.checkpoint_dir, 'pretrained_for_init_ablation.pth')
    
    if not os.path.exists(pretrained_path):
        model_pretrained = DisagreementPredictor()
        model_pretrained, _ = pretrain_on_hard_labels(
            model_pretrained, pretrain_loader, num_epochs=args.pretrain_epochs,
            device=args.device, save_path=pretrained_path
        )
    else:
        model_pretrained = DisagreementPredictor()
        model_pretrained.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    
    save_path = os.path.join(args.checkpoint_dir, 'init_ablation_cifar10.pth')
    model_pretrained, _ = finetune_on_soft_labels(
        model_pretrained, train_loader, val_loader, kl_divergence_loss, 'cifar10',
        num_epochs=args.finetune_epochs, device=args.device, save_path=save_path
    )
    models['cifar10_pretrained'] = model_pretrained
    logger.info("✓ CIFAR-10 pretrained model trained")
    
    # Compare models
    logger.info("\nComparing initialization strategies...")
    comparison_df = compare_backbone_initialization(
        models, test_loader, device=args.device,
        output_dir=os.path.join(args.output_dir, 'initialization')
    )
    
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info("\n✓ Initialization ablation complete")
    
    return comparison_df


def run_training_strategy_ablation(pretrain_loader, train_loader, val_loader, test_loader, args, logger):
    """Ablation study: Compare training strategies."""
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY: Training Strategies")
    logger.info("="*70)
    
    models = {}
    
    # Strategy 1: Two-stage (pretrain + finetune)
    logger.info("\nTraining with two-stage strategy...")
    pretrained_path = os.path.join(args.checkpoint_dir, 'pretrained_for_strategy_ablation.pth')
    
    if not os.path.exists(pretrained_path):
        model_two_stage = DisagreementPredictor()
        model_two_stage, _ = pretrain_on_hard_labels(
            model_two_stage, pretrain_loader, num_epochs=args.pretrain_epochs,
            device=args.device, save_path=pretrained_path
        )
    else:
        model_two_stage = DisagreementPredictor()
        model_two_stage.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    
    save_path = os.path.join(args.checkpoint_dir, 'strategy_ablation_two_stage.pth')
    model_two_stage, _ = finetune_on_soft_labels(
        model_two_stage, train_loader, val_loader, kl_divergence_loss, 'two_stage',
        num_epochs=args.finetune_epochs, device=args.device, save_path=save_path
    )
    models['two_stage'] = model_two_stage
    logger.info("✓ Two-stage model trained")
    
    # Strategy 2: Single-stage (finetune only)
    logger.info("\nTraining with single-stage strategy...")
    model_single_stage = DisagreementPredictor()
    save_path = os.path.join(args.checkpoint_dir, 'strategy_ablation_single_stage.pth')
    model_single_stage, _ = finetune_on_soft_labels(
        model_single_stage, train_loader, val_loader, kl_divergence_loss, 'single_stage',
        num_epochs=args.finetune_epochs, device=args.device, save_path=save_path
    )
    models['single_stage'] = model_single_stage
    logger.info("✓ Single-stage model trained")
    
    # Compare models
    logger.info("\nComparing training strategies...")
    comparison_df = compare_training_strategies(
        models, test_loader, device=args.device,
        output_dir=os.path.join(args.output_dir, 'training_strategy')
    )
    
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info("\n✓ Training strategy ablation complete")
    
    return comparison_df


def run_architecture_ablation(pretrain_loader, train_loader, val_loader, test_loader, args, logger):
    """Ablation study: Compare prediction head architectures."""
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY: Prediction Head Architectures")
    logger.info("="*70)
    
    models = {}
    
    # Pretrain backbone once
    logger.info("\nPretraining backbone...")
    pretrained_path = os.path.join(args.checkpoint_dir, 'pretrained_for_arch_ablation.pth')
    
    if not os.path.exists(pretrained_path):
        model = DisagreementPredictor()
        model, _ = pretrain_on_hard_labels(
            model, pretrain_loader, num_epochs=args.pretrain_epochs,
            device=args.device, save_path=pretrained_path
        )
    
    # Architecture 1: Single linear layer (512 → 10)
    logger.info("\nTraining with single linear layer head...")
    
    class SingleLayerPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            from src.model import create_modified_resnet18
            self.backbone = create_modified_resnet18()
            self.head = nn.Sequential(
                nn.Linear(512, 10),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    model_single = SingleLayerPredictor()
    # Load pretrained backbone weights
    pretrained_state = torch.load(pretrained_path, map_location='cpu')
    backbone_state = {k.replace('backbone.', ''): v for k, v in pretrained_state.items() if k.startswith('backbone.')}
    model_single.backbone.load_state_dict(backbone_state, strict=False)
    
    save_path = os.path.join(args.checkpoint_dir, 'arch_ablation_single_layer.pth')
    model_single, _ = finetune_on_soft_labels(
        model_single, train_loader, val_loader, kl_divergence_loss, 'single_layer',
        num_epochs=args.finetune_epochs, device=args.device, save_path=save_path
    )
    models['single_layer'] = model_single
    logger.info("✓ Single layer model trained")
    
    # Architecture 2: Two-layer MLP (512 → 256 → 10)
    logger.info("\nTraining with two-layer MLP head...")
    model_mlp = DisagreementPredictor()
    model_mlp.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    
    save_path = os.path.join(args.checkpoint_dir, 'arch_ablation_two_layer_mlp.pth')
    model_mlp, _ = finetune_on_soft_labels(
        model_mlp, train_loader, val_loader, kl_divergence_loss, 'two_layer_mlp',
        num_epochs=args.finetune_epochs, device=args.device, save_path=save_path
    )
    models['two_layer_mlp'] = model_mlp
    logger.info("✓ Two-layer MLP model trained")
    
    # Compare models
    logger.info("\nComparing prediction head architectures...")
    comparison_df = compare_prediction_head_architectures(
        models, test_loader, device=args.device,
        output_dir=os.path.join(args.output_dir, 'architecture')
    )
    
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info("\n✓ Architecture ablation complete")
    
    return comparison_df


def main():
    """Main ablation study pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("CIFAR-10 Human Disagreement Predictor - Ablation Studies")
    logger.info("="*70)
    logger.info(f"Device: {args.device}")
    logger.info(f"Random seed: {args.seed}")
    
    # Handle 'all' option
    if 'all' in args.studies:
        args.studies = ['loss', 'initialization', 'training_strategy', 'architecture']
    
    logger.info(f"Running ablation studies: {', '.join(args.studies)}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Prepare data
    pretrain_loader, train_loader, val_loader, test_loader = prepare_data(args, logger)
    
    # Run ablation studies
    results = {}
    
    if 'loss' in args.studies:
        results['loss_functions'] = run_loss_function_ablation(
            pretrain_loader, train_loader, val_loader, test_loader, args, logger
        )
    
    if 'initialization' in args.studies:
        results['initialization'] = run_initialization_ablation(
            pretrain_loader, train_loader, val_loader, test_loader, args, logger
        )
    
    if 'training_strategy' in args.studies:
        results['training_strategy'] = run_training_strategy_ablation(
            pretrain_loader, train_loader, val_loader, test_loader, args, logger
        )
    
    if 'architecture' in args.studies:
        results['architecture'] = run_architecture_ablation(
            pretrain_loader, train_loader, val_loader, test_loader, args, logger
        )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDIES COMPLETE")
    logger.info("="*70)
    logger.info(f"\nCompleted {len(results)} ablation study/studies:")
    for study_name in results.keys():
        logger.info(f"  - {study_name}")
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
