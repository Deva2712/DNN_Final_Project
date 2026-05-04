#!/usr/bin/env python3
"""
Evaluation Script

Evaluates trained disagreement prediction models on the test set.
Generates comprehensive metrics, visualizations, and exports results.

Usage:
    python evaluate.py [options]

Examples:
    # Evaluate all three models
    python evaluate.py

    # Evaluate specific model
    python evaluate.py --model-path checkpoints/finetuned_kl_best.pth

    # Custom output directory
    python evaluate.py --output-dir ./my_evaluation_results

    # Generate all visualizations
    python evaluate.py --generate-visualizations
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DisagreementPredictor
from src.data_pipeline import (
    load_cifar10_data,
    load_cifar10h_data,
    compute_soft_labels,
    align_datasets,
    split_dataset,
    compute_entropy,
    CIFAR10HDataset
)
from src.training import set_seed, get_test_transform
from src.evaluation import (
    evaluate_model,
    compute_distribution_metrics,
    compute_entropy_correlation,
    compute_precision_at_k,
    analyze_per_class_performance,
    evaluate_corruption_robustness,
    EvaluationMetrics
)
from src.visualization import (
    visualize_gradcam_comparison,
    visualize_failure_cases,
    plot_corruption_robustness
)
from src.logging_config import setup_logging
import matplotlib.pyplot as plt
import numpy as np


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate CIFAR-10 disagreement prediction models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to specific model checkpoint (if None, evaluates all models in checkpoint-dir)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory containing model checkpoints'
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
        default='./outputs/evaluation_results',
        help='Directory to save evaluation results'
    )
    
    # Evaluation options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--generate-visualizations',
        action='store_true',
        help='Generate Grad-CAM and failure case visualizations'
    )
    parser.add_argument(
        '--evaluate-robustness',
        action='store_true',
        help='Evaluate robustness to image corruptions'
    )
    parser.add_argument(
        '--num-failure-cases',
        type=int,
        default=10,
        help='Number of failure cases to visualize'
    )
    
    # Device and seed
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for evaluation'
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


def prepare_test_data(args, logger):
    """Prepare test dataset and dataloader."""
    logger.info("="*70)
    logger.info("Preparing test dataset")
    logger.info("="*70)
    
    # Load CIFAR-10 test set
    cifar10_images, cifar10_labels = load_cifar10_data(
        data_dir=args.cifar10_dir,
        train=False,
        download=True
    )
    
    # Load CIFAR-10H
    cifar10h_counts, cifar10h_probs = load_cifar10h_data(data_dir=args.cifar10h_dir)
    soft_labels = compute_soft_labels(cifar10h_counts)
    
    # Align and split
    aligned_data = align_datasets(cifar10_images, cifar10_labels, soft_labels)
    _, _, test_data = split_dataset(aligned_data, random_seed=args.seed)
    
    # Compute entropies
    entropies = compute_entropy(soft_labels)
    
    # Create test dataset
    import numpy as np
    images = torch.stack([torch.from_numpy(img).float() / 255.0 for img, _, _ in test_data])
    soft_labels_tensor = torch.stack([torch.from_numpy(sl).float() for _, sl, _ in test_data])
    hard_labels_tensor = torch.tensor([hl for _, _, hl in test_data], dtype=torch.long)
    
    # Get entropies for test samples
    indices = []
    for img, _, _ in test_data:
        idx = np.where((cifar10_images == img).all(axis=(1, 2, 3)))[0][0]
        indices.append(idx)
    entropies_tensor = torch.from_numpy(entropies[indices]).float()
    
    test_dataset = CIFAR10HDataset(
        images, soft_labels_tensor, hard_labels_tensor, entropies_tensor,
        transform=get_test_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    logger.info(f"✓ Created test dataloader: {len(test_dataset)} samples")
    
    return test_loader, test_dataset


def evaluate_single_model(model_path, model_name, test_loader, test_dataset, args, logger):
    """Evaluate a single model."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"{'='*70}")
    
    # Load model
    model = DisagreementPredictor()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info(f"✓ Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Create output directory for this model
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Evaluate model
    logger.info("\nComputing evaluation metrics...")
    try:
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=args.device,
            output_dir=model_output_dir
        )
        logger.info("✓ Evaluation metrics computed")
        
        # Print metrics
        logger.info(f"\nDistribution Matching Metrics:")
        logger.info(f"  KL Divergence: {metrics['mean_kl']:.4f} ± {metrics['std_kl']:.4f}")
        logger.info(f"  JS Divergence: {metrics['mean_js']:.4f} ± {metrics['std_js']:.4f}")
        logger.info(f"  Cosine Similarity: {metrics['mean_cosine']:.4f} ± {metrics['std_cosine']:.4f}")
        
        logger.info(f"\nEntropy Prediction Quality:")
        logger.info(f"  Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
        logger.info(f"  Spearman ρ: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")
        
        logger.info(f"\nPrecision@K:")
        logger.info(f"  Precision@100: {metrics['precision@100']:.4f}")
        logger.info(f"  Precision@200: {metrics['precision@200']:.4f}")
        logger.info(f"  Precision@500: {metrics['precision@500']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        return None
    
    # Per-class analysis
    logger.info("\nAnalyzing per-class performance...")
    try:
        per_class_df = analyze_per_class_performance(
            model=model,
            test_loader=test_loader,
            class_names=CIFAR10_CLASSES,
            device=args.device,
            output_dir=model_output_dir
        )
        logger.info("✓ Per-class analysis complete")
    except Exception as e:
        logger.warning(f"Per-class analysis failed: {e}")
    
    # Generate visualizations
    if args.generate_visualizations:
        logger.info("\nGenerating visualizations...")
        
        # Grad-CAM visualization
        try:
            # Select low and high entropy images
            entropies = test_dataset.entropies.numpy()
            low_entropy_indices = np.argsort(entropies)[:5]
            high_entropy_indices = np.argsort(entropies)[-5:]
            
            low_entropy_images = torch.stack([test_dataset.images[i] for i in low_entropy_indices])
            high_entropy_images = torch.stack([test_dataset.images[i] for i in high_entropy_indices])
            
            gradcam_path = os.path.join(model_output_dir, 'gradcam_comparison.png')
            visualize_gradcam_comparison(
                model=model,
                low_entropy_images=low_entropy_images,
                high_entropy_images=high_entropy_images,
                save_path=gradcam_path,
                device=args.device
            )
            logger.info(f"✓ Saved Grad-CAM visualization to {gradcam_path}")
        except Exception as e:
            logger.warning(f"Grad-CAM visualization failed: {e}")
        
        # Failure case visualization
        try:
            failure_path = os.path.join(model_output_dir, 'failure_cases.png')
            visualize_failure_cases(
                model=model,
                test_loader=test_loader,
                num_cases=args.num_failure_cases,
                save_path=failure_path,
                device=args.device,
                class_names=CIFAR10_CLASSES
            )
            logger.info(f"✓ Saved failure case visualization to {failure_path}")
        except Exception as e:
            logger.warning(f"Failure case visualization failed: {e}")
        
        # Entropy correlation scatter plot
        try:
            entropy_metrics = compute_entropy_correlation(model, test_loader, args.device)
            
            plt.figure(figsize=(8, 8))
            plt.scatter(entropy_metrics['true_entropies'], entropy_metrics['pred_entropies'], 
                       alpha=0.5, s=20)
            plt.plot([0, 3.5], [0, 3.5], 'r--', label='Perfect prediction')
            plt.xlabel('True Entropy (bits)', fontsize=12)
            plt.ylabel('Predicted Entropy (bits)', fontsize=12)
            plt.title(f'Entropy Prediction Quality\n'
                     f'Pearson r={entropy_metrics["pearson_r"]:.3f}, '
                     f'Spearman ρ={entropy_metrics["spearman_r"]:.3f}',
                     fontsize=13)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            scatter_path = os.path.join(model_output_dir, 'entropy_correlation.png')
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved entropy correlation plot to {scatter_path}")
        except Exception as e:
            logger.warning(f"Entropy correlation plot failed: {e}")
    
    # Evaluate robustness
    if args.evaluate_robustness:
        logger.info("\nEvaluating robustness to corruptions...")
        try:
            robustness_results = evaluate_corruption_robustness(
                model=model,
                test_loader=test_loader,
                device=args.device,
                output_dir=model_output_dir
            )
            
            # Plot robustness results
            robustness_plot_path = os.path.join(model_output_dir, 'corruption_robustness.png')
            plot_corruption_robustness(robustness_results, robustness_plot_path)
            logger.info(f"✓ Saved robustness plot to {robustness_plot_path}")
        except Exception as e:
            logger.warning(f"Robustness evaluation failed: {e}")
    
    logger.info(f"\n✓ Evaluation complete for {model_name}")
    logger.info(f"  Results saved to: {model_output_dir}")
    
    return metrics


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("CIFAR-10 Human Disagreement Predictor - Evaluation")
    logger.info("="*70)
    logger.info(f"Device: {args.device}")
    logger.info(f"Random seed: {args.seed}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare test data
    test_loader, test_dataset = prepare_test_data(args, logger)
    
    # Determine which models to evaluate
    if args.model_path is not None:
        # Evaluate single model
        model_paths = [(args.model_path, os.path.basename(args.model_path).replace('.pth', ''))]
    else:
        # Evaluate all models in checkpoint directory
        import glob
        model_files = glob.glob(os.path.join(args.checkpoint_dir, 'finetuned_*_best.pth'))
        if not model_files:
            logger.error(f"No model checkpoints found in {args.checkpoint_dir}")
            sys.exit(1)
        model_paths = [(path, os.path.basename(path).replace('.pth', '')) for path in model_files]
    
    logger.info(f"\nFound {len(model_paths)} model(s) to evaluate:")
    for path, name in model_paths:
        logger.info(f"  - {name}")
    
    # Evaluate each model
    all_results = {}
    
    for model_path, model_name in model_paths:
        metrics = evaluate_single_model(
            model_path=model_path,
            model_name=model_name,
            test_loader=test_loader,
            test_dataset=test_dataset,
            args=args,
            logger=logger
        )
        
        if metrics is not None:
            all_results[model_name] = metrics
    
    # Generate comparison table if multiple models
    if len(all_results) > 1:
        logger.info("\n" + "="*70)
        logger.info("Generating comparison table")
        logger.info("="*70)
        
        try:
            # Create comparison DataFrame
            comparison_data = []
            for model_name, metrics in all_results.items():
                row = {
                    'model': model_name,
                    'mean_kl': metrics['mean_kl'],
                    'mean_js': metrics['mean_js'],
                    'mean_cosine': metrics['mean_cosine'],
                    'pearson_r': metrics['pearson_r'],
                    'spearman_r': metrics['spearman_r'],
                    'precision@100': metrics['precision@100'],
                    'precision@200': metrics['precision@200'],
                    'precision@500': metrics['precision@500']
                }
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Save to CSV
            comparison_path = os.path.join(args.output_dir, 'model_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"✓ Saved comparison table to {comparison_path}")
            
            # Print comparison table
            logger.info("\nModel Comparison:")
            logger.info("\n" + comparison_df.to_string(index=False))
            
        except Exception as e:
            logger.warning(f"Failed to generate comparison table: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nEvaluated {len(all_results)} model(s)")
    logger.info(f"Results saved to: {args.output_dir}")
    
    if all_results:
        logger.info("\nBest performing model by metric:")
        
        # Find best model for each metric
        best_kl = min(all_results.items(), key=lambda x: x[1]['mean_kl'])
        best_pearson = max(all_results.items(), key=lambda x: x[1]['pearson_r'])
        best_precision = max(all_results.items(), key=lambda x: x[1]['precision@100'])
        
        logger.info(f"  Lowest KL divergence: {best_kl[0]} ({best_kl[1]['mean_kl']:.4f})")
        logger.info(f"  Highest Pearson r: {best_pearson[0]} ({best_pearson[1]['pearson_r']:.4f})")
        logger.info(f"  Highest Precision@100: {best_precision[0]} ({best_precision[1]['precision@100']:.4f})")
    
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
