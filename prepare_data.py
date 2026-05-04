#!/usr/bin/env python3
"""
Data Preparation Script

Downloads and prepares CIFAR-10 and CIFAR-10H datasets, generates visualizations,
and saves dataset splits and configurations.

Usage:
    python prepare_data.py [options]

Examples:
    # Basic usage with default settings
    python prepare_data.py

    # Specify custom data directories
    python prepare_data.py --cifar10-dir ./my_data --cifar10h-dir ./cifar10h_data

    # Custom output directory
    python prepare_data.py --output-dir ./my_outputs

    # Custom split sizes
    python prepare_data.py --train-size 5000 --val-size 2500 --test-size 2500
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import (
    load_cifar10_data,
    load_cifar10h_data,
    compute_soft_labels,
    align_datasets,
    split_dataset,
    compute_entropy,
    CIFAR10HDataset,
    DataPipelineConfig
)
from src.visualization import (
    plot_entropy_histogram,
    plot_per_class_entropy,
    plot_example_grid
)
from src.logging_config import setup_logging


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare CIFAR-10 and CIFAR-10H datasets for disagreement prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data directories
    parser.add_argument(
        '--cifar10-dir',
        type=str,
        default='./data',
        help='Directory to store/load CIFAR-10 dataset'
    )
    parser.add_argument(
        '--cifar10h-dir',
        type=str,
        default='./cifar-10h-1.0.0/data',
        help='Directory containing CIFAR-10H dataset files'
    )
    
    # Output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/data_visualizations',
        help='Directory to save visualizations and configurations'
    )
    
    # Split sizes
    parser.add_argument(
        '--train-size',
        type=int,
        default=6000,
        help='Number of training samples'
    )
    parser.add_argument(
        '--val-size',
        type=int,
        default=2000,
        help='Number of validation samples'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=2000,
        help='Number of test samples'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Download option
    parser.add_argument(
        '--download',
        action='store_true',
        default=True,
        help='Download CIFAR-10 if not present'
    )
    parser.add_argument(
        '--no-download',
        action='store_false',
        dest='download',
        help='Do not download CIFAR-10 (fail if not present)'
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


def main():
    """Main data preparation pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("CIFAR-10 Human Disagreement Predictor - Data Preparation")
    logger.info("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate split sizes
    total_size = args.train_size + args.val_size + args.test_size
    if total_size != 10000:
        logger.error(f"Split sizes must sum to 10000, got {total_size}")
        sys.exit(1)
    
    # Step 1: Load CIFAR-10 test set
    logger.info("\n" + "="*70)
    logger.info("Step 1: Loading CIFAR-10 test set")
    logger.info("="*70)
    
    try:
        cifar10_images, cifar10_labels = load_cifar10_data(
            data_dir=args.cifar10_dir,
            train=False,
            download=args.download
        )
        logger.info(f"✓ Loaded {len(cifar10_images)} CIFAR-10 test images")
    except Exception as e:
        logger.error(f"Failed to load CIFAR-10: {e}")
        sys.exit(1)
    
    # Step 2: Load CIFAR-10H dataset
    logger.info("\n" + "="*70)
    logger.info("Step 2: Loading CIFAR-10H dataset")
    logger.info("="*70)
    
    try:
        cifar10h_counts, cifar10h_probs = load_cifar10h_data(data_dir=args.cifar10h_dir)
        logger.info(f"✓ Loaded CIFAR-10H data: {cifar10h_probs.shape}")
    except Exception as e:
        logger.error(f"Failed to load CIFAR-10H: {e}")
        logger.error("Please download CIFAR-10H from https://github.com/jcpeterson/cifar-10h")
        sys.exit(1)
    
    # Step 3: Compute soft labels
    logger.info("\n" + "="*70)
    logger.info("Step 3: Computing soft labels from annotator counts")
    logger.info("="*70)
    
    try:
        soft_labels = compute_soft_labels(cifar10h_counts)
        logger.info(f"✓ Computed {len(soft_labels)} soft label distributions")
    except Exception as e:
        logger.error(f"Failed to compute soft labels: {e}")
        sys.exit(1)
    
    # Step 4: Align datasets
    logger.info("\n" + "="*70)
    logger.info("Step 4: Aligning CIFAR-10H with CIFAR-10 test set")
    logger.info("="*70)
    
    try:
        aligned_data = align_datasets(
            cifar10_images,
            cifar10_labels,
            soft_labels
        )
        logger.info(f"✓ Aligned {len(aligned_data)} image-label pairs")
    except Exception as e:
        logger.error(f"Failed to align datasets: {e}")
        sys.exit(1)
    
    # Step 5: Compute entropy
    logger.info("\n" + "="*70)
    logger.info("Step 5: Computing Shannon entropy for all distributions")
    logger.info("="*70)
    
    try:
        entropies = compute_entropy(soft_labels)
        logger.info(f"✓ Computed entropy values")
        logger.info(f"  Min entropy: {entropies.min():.3f} bits")
        logger.info(f"  Max entropy: {entropies.max():.3f} bits")
        logger.info(f"  Mean entropy: {entropies.mean():.3f} bits")
        logger.info(f"  Std entropy: {entropies.std():.3f} bits")
    except Exception as e:
        logger.error(f"Failed to compute entropy: {e}")
        sys.exit(1)
    
    # Step 6: Split dataset
    logger.info("\n" + "="*70)
    logger.info("Step 6: Splitting dataset into train/val/test")
    logger.info("="*70)
    
    try:
        train_data, val_data, test_data = split_dataset(aligned_data, random_seed=args.seed)
        logger.info(f"✓ Split dataset:")
        logger.info(f"  Training: {len(train_data)} samples")
        logger.info(f"  Validation: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        sys.exit(1)
    
    # Step 7: Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("Step 7: Generating data visualizations")
    logger.info("="*70)
    
    # Entropy histogram
    try:
        histogram_path = os.path.join(args.output_dir, 'entropy_histogram.png')
        plot_entropy_histogram(entropies, histogram_path)
        logger.info(f"✓ Saved entropy histogram to {histogram_path}")
    except Exception as e:
        logger.warning(f"Failed to generate entropy histogram: {e}")
    
    # Per-class entropy distribution
    try:
        per_class_path = os.path.join(args.output_dir, 'per_class_entropy.png')
        plot_per_class_entropy(entropies, cifar10_labels, CIFAR10_CLASSES, per_class_path)
        logger.info(f"✓ Saved per-class entropy plot to {per_class_path}")
    except Exception as e:
        logger.warning(f"Failed to generate per-class entropy plot: {e}")
    
    # Example image grid
    try:
        example_grid_path = os.path.join(args.output_dir, 'example_grid.png')
        plot_example_grid(cifar10_images, entropies, soft_labels, example_grid_path, CIFAR10_CLASSES)
        logger.info(f"✓ Saved example grid to {example_grid_path}")
    except Exception as e:
        logger.warning(f"Failed to generate example grid: {e}")
    
    # Step 8: Save configuration
    logger.info("\n" + "="*70)
    logger.info("Step 8: Saving data pipeline configuration")
    logger.info("="*70)
    
    try:
        config = DataPipelineConfig(
            cifar10_data_dir=args.cifar10_dir,
            cifar10h_data_dir=args.cifar10h_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            random_seed=args.seed
        )
        config_path = os.path.join(args.output_dir, 'data_pipeline_config.json')
        config.to_json(config_path)
        logger.info(f"✓ Saved configuration to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save configuration: {e}")
    
    # Step 9: Save dataset splits (optional - for later use)
    logger.info("\n" + "="*70)
    logger.info("Step 9: Saving dataset split indices")
    logger.info("="*70)
    
    try:
        # Extract indices from aligned_data
        train_indices = [i for i, _ in enumerate(aligned_data) if any((aligned_data[i][0] == td[0]).all() for td in train_data)]
        val_indices = [i for i, _ in enumerate(aligned_data) if any((aligned_data[i][0] == vd[0]).all() for vd in val_data)]
        test_indices = [i for i, _ in enumerate(aligned_data) if any((aligned_data[i][0] == td[0]).all() for td in test_data)]
        
        # Note: The above is inefficient for large datasets. For production, use a better approach.
        # For now, we'll just save the split sizes as they're deterministic with the seed.
        
        split_info = {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'random_seed': args.seed,
            'note': 'Use the same random seed to reproduce these splits'
        }
        
        import json
        split_path = os.path.join(args.output_dir, 'split_info.json')
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        logger.info(f"✓ Saved split information to {split_path}")
    except Exception as e:
        logger.warning(f"Failed to save split indices: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total images: {len(aligned_data)}")
    logger.info(f"  Training: {len(train_data)} ({100*len(train_data)/len(aligned_data):.1f}%)")
    logger.info(f"  Validation: {len(val_data)} ({100*len(val_data)/len(aligned_data):.1f}%)")
    logger.info(f"  Test: {len(test_data)} ({100*len(test_data)/len(aligned_data):.1f}%)")
    logger.info(f"\nEntropy Statistics:")
    logger.info(f"  Range: [{entropies.min():.3f}, {entropies.max():.3f}] bits")
    logger.info(f"  Mean ± Std: {entropies.mean():.3f} ± {entropies.std():.3f} bits")
    logger.info(f"\nOutputs saved to: {args.output_dir}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
