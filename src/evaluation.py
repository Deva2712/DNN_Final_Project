"""
Evaluation Module

Implements comprehensive evaluation metrics including distribution matching,
entropy prediction quality, Precision@K, and ablation studies.
"""

import logging
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from src.losses import kl_divergence_loss, js_divergence_loss, compute_entropy

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Dataclass to store comprehensive evaluation metrics.
    
    Attributes:
        mean_kl: Mean KL divergence
        std_kl: Standard deviation of KL divergence
        mean_js: Mean JS divergence
        std_js: Standard deviation of JS divergence
        mean_cosine: Mean cosine similarity
        std_cosine: Standard deviation of cosine similarity
        pearson_r: Pearson correlation coefficient for entropy
        pearson_p: P-value for Pearson correlation
        spearman_r: Spearman correlation coefficient for entropy
        spearman_p: P-value for Spearman correlation
        precision_at_100: Precision@100 for ambiguous image identification
        precision_at_200: Precision@200 for ambiguous image identification
        precision_at_500: Precision@500 for ambiguous image identification
    """
    mean_kl: float
    std_kl: float
    mean_js: float
    std_js: float
    mean_cosine: float
    std_cosine: float
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    precision_at_100: float
    precision_at_200: float
    precision_at_500: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation metrics to {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EvaluationMetrics':
        """Create from dictionary."""
        # Handle both formats (with and without @ in precision keys)
        precision_100 = data.get('precision@100', data.get('precision_at_100', 0.0))
        precision_200 = data.get('precision@200', data.get('precision_at_200', 0.0))
        precision_500 = data.get('precision@500', data.get('precision_at_500', 0.0))
        
        return cls(
            mean_kl=data['mean_kl'],
            std_kl=data['std_kl'],
            mean_js=data['mean_js'],
            std_js=data['std_js'],
            mean_cosine=data['mean_cosine'],
            std_cosine=data['std_cosine'],
            pearson_r=data['pearson_r'],
            pearson_p=data['pearson_p'],
            spearman_r=data['spearman_r'],
            spearman_p=data['spearman_p'],
            precision_at_100=precision_100,
            precision_at_200=precision_200,
            precision_at_500=precision_500
        )



def compute_distribution_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute distribution matching metrics (KL, JS, cosine similarity).
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
    
    Returns:
        metrics: Dictionary with mean and std for KL, JS, and cosine similarity
    """
    logger.info("Computing distribution matching metrics")
    
    model.eval()
    model = model.to(device)
    
    kl_values = []
    js_values = []
    cosine_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (image, soft_label, hard_label, entropy)
            images, soft_labels, _, _ = batch
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            
            # Get predictions
            pred_probs = model(images)
            
            # Compute KL divergence for each sample
            epsilon = 1e-7
            pred_probs_safe = pred_probs + epsilon
            soft_labels_safe = soft_labels + epsilon
            pred_probs_safe = pred_probs_safe / pred_probs_safe.sum(dim=1, keepdim=True)
            soft_labels_safe = soft_labels_safe / soft_labels_safe.sum(dim=1, keepdim=True)
            
            kl_per_sample = (soft_labels_safe * torch.log(soft_labels_safe / pred_probs_safe)).sum(dim=1)
            kl_values.extend(kl_per_sample.cpu().numpy())
            
            # Compute JS divergence for each sample
            m = 0.5 * (pred_probs_safe + soft_labels_safe)
            kl_target_m = (soft_labels_safe * torch.log(soft_labels_safe / m)).sum(dim=1)
            kl_pred_m = (pred_probs_safe * torch.log(pred_probs_safe / m)).sum(dim=1)
            js_per_sample = 0.5 * kl_target_m + 0.5 * kl_pred_m
            js_values.extend(js_per_sample.cpu().numpy())
            
            # Compute cosine similarity for each sample
            cosine_per_sample = F.cosine_similarity(pred_probs, soft_labels, dim=1)
            cosine_values.extend(cosine_per_sample.cpu().numpy())
    
    # Convert to numpy arrays
    kl_values = np.array(kl_values)
    js_values = np.array(js_values)
    cosine_values = np.array(cosine_values)
    
    metrics = {
        'mean_kl': float(np.mean(kl_values)),
        'std_kl': float(np.std(kl_values)),
        'mean_js': float(np.mean(js_values)),
        'std_js': float(np.std(js_values)),
        'mean_cosine': float(np.mean(cosine_values)),
        'std_cosine': float(np.std(cosine_values))
    }
    
    logger.info(f"Distribution metrics - KL: {metrics['mean_kl']:.4f}±{metrics['std_kl']:.4f}, "
                f"JS: {metrics['mean_js']:.4f}±{metrics['std_js']:.4f}, "
                f"Cosine: {metrics['mean_cosine']:.4f}±{metrics['std_cosine']:.4f}")
    
    return metrics


def compute_entropy_correlation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute entropy prediction quality metrics (Pearson, Spearman correlation).
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
    
    Returns:
        metrics: Dictionary with Pearson and Spearman correlation coefficients
    """
    logger.info("Computing entropy correlation metrics")
    
    model.eval()
    model = model.to(device)
    
    true_entropies = []
    pred_entropies = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (image, soft_label, hard_label, entropy)
            images, soft_labels, _, true_entropy = batch
            images = images.to(device)
            
            # Get predictions
            pred_probs = model(images)
            
            # Compute predicted entropy
            pred_entropy = compute_entropy(pred_probs)
            
            # Store values
            true_entropies.extend(true_entropy.cpu().numpy())
            pred_entropies.extend(pred_entropy.cpu().numpy())
    
    # Convert to numpy arrays
    true_entropies = np.array(true_entropies)
    pred_entropies = np.array(pred_entropies)
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(true_entropies, pred_entropies)
    spearman_r, spearman_p = spearmanr(true_entropies, pred_entropies)
    
    metrics = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'true_entropies': true_entropies,  # Store for visualization
        'pred_entropies': pred_entropies   # Store for visualization
    }
    
    logger.info(f"Entropy correlation - Pearson r: {pearson_r:.4f} (p={pearson_p:.4e}), "
                f"Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4e})")
    
    return metrics


def compute_precision_at_k(
    model: nn.Module,
    test_loader: DataLoader,
    k_values: List[int] = [100, 200, 500],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute Precision@K for identifying ambiguous images.
    
    Precision@K = |top-K by true entropy ∩ top-K by pred entropy| / K
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        k_values: List of K values to compute precision for
        device: 'cuda' or 'cpu'
    
    Returns:
        metrics: Dictionary with Precision@K for each K value
    """
    logger.info(f"Computing Precision@K for K={k_values}")
    
    model.eval()
    model = model.to(device)
    
    true_entropies = []
    pred_entropies = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (image, soft_label, hard_label, entropy)
            images, soft_labels, _, true_entropy = batch
            images = images.to(device)
            
            # Get predictions
            pred_probs = model(images)
            
            # Compute predicted entropy
            pred_entropy = compute_entropy(pred_probs)
            
            # Store values
            true_entropies.extend(true_entropy.cpu().numpy())
            pred_entropies.extend(pred_entropy.cpu().numpy())
    
    # Convert to numpy arrays
    true_entropies = np.array(true_entropies)
    pred_entropies = np.array(pred_entropies)
    
    # Rank images by entropy (descending order - highest entropy first)
    true_ranking = np.argsort(true_entropies)[::-1]
    pred_ranking = np.argsort(pred_entropies)[::-1]
    
    metrics = {}
    for k in k_values:
        # Get top-K indices
        true_top_k = set(true_ranking[:k])
        pred_top_k = set(pred_ranking[:k])
        
        # Compute overlap
        overlap = len(true_top_k & pred_top_k)
        precision = overlap / k
        
        metrics[f'precision@{k}'] = float(precision)
        logger.info(f"Precision@{k}: {precision:.4f} (overlap: {overlap}/{k})")
    
    return metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation combining all metrics.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        output_dir: Optional directory to save metrics JSON
    
    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    logger.info("Running comprehensive model evaluation")
    
    # Compute all metrics
    dist_metrics = compute_distribution_metrics(model, test_loader, device)
    entropy_metrics = compute_entropy_correlation(model, test_loader, device)
    precision_metrics = compute_precision_at_k(model, test_loader, device=device)
    
    # Combine all metrics (exclude arrays for JSON serialization)
    all_metrics = {
        **dist_metrics,
        'pearson_r': entropy_metrics['pearson_r'],
        'pearson_p': entropy_metrics['pearson_p'],
        'spearman_r': entropy_metrics['spearman_r'],
        'spearman_p': entropy_metrics['spearman_p'],
        **precision_metrics
    }
    
    # Save to JSON if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to {output_path}")
    
    logger.info("Comprehensive evaluation complete")
    return all_metrics


def compare_loss_functions(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare models trained with different loss functions.
    
    Args:
        models: Dictionary mapping loss function name to trained model
                e.g., {'kl': model_kl, 'js': model_js, 'custom': model_custom}
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        output_dir: Optional directory to save comparison table
    
    Returns:
        comparison_df: DataFrame with metrics for each loss function
    """
    logger.info("Comparing loss functions")
    
    results = []
    
    for loss_name, model in models.items():
        logger.info(f"Evaluating model trained with {loss_name} loss")
        
        # Compute all metrics
        metrics = evaluate_model(model, test_loader, device, output_dir=None)
        
        # Add loss function name
        metrics['loss_function'] = loss_name
        results.append(metrics)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns to put loss_function first
    cols = ['loss_function'] + [col for col in comparison_df.columns if col != 'loss_function']
    comparison_df = comparison_df[cols]
    
    # Save to CSV if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'loss_function_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Saved loss function comparison to {output_path}")
    
    logger.info("Loss function comparison complete")
    return comparison_df


def compare_backbone_initialization(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare models with different backbone initialization strategies.
    
    Args:
        models: Dictionary mapping initialization strategy to trained model
                e.g., {'random': model_random, 'cifar10': model_cifar10, 'imagenet': model_imagenet}
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        output_dir: Optional directory to save comparison table
    
    Returns:
        comparison_df: DataFrame with metrics for each initialization strategy
    """
    logger.info("Comparing backbone initialization strategies")
    
    results = []
    
    for init_name, model in models.items():
        logger.info(f"Evaluating model with {init_name} initialization")
        
        # Compute all metrics
        metrics = evaluate_model(model, test_loader, device, output_dir=None)
        
        # Add initialization strategy name
        metrics['initialization'] = init_name
        results.append(metrics)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns to put initialization first
    cols = ['initialization'] + [col for col in comparison_df.columns if col != 'initialization']
    comparison_df = comparison_df[cols]
    
    # Save to CSV if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'initialization_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Saved initialization comparison to {output_path}")
    
    logger.info("Backbone initialization comparison complete")
    return comparison_df


def compare_training_strategies(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare models trained with different training strategies.
    
    Args:
        models: Dictionary mapping training strategy to trained model
                e.g., {'two_stage': model_two_stage, 'single_stage': model_single_stage}
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        output_dir: Optional directory to save comparison table
    
    Returns:
        comparison_df: DataFrame with metrics for each training strategy
    """
    logger.info("Comparing training strategies")
    
    results = []
    
    for strategy_name, model in models.items():
        logger.info(f"Evaluating model trained with {strategy_name} strategy")
        
        # Compute all metrics
        metrics = evaluate_model(model, test_loader, device, output_dir=None)
        
        # Add training strategy name
        metrics['training_strategy'] = strategy_name
        results.append(metrics)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns to put training_strategy first
    cols = ['training_strategy'] + [col for col in comparison_df.columns if col != 'training_strategy']
    comparison_df = comparison_df[cols]
    
    # Save to CSV if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_strategy_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Saved training strategy comparison to {output_path}")
    
    logger.info("Training strategy comparison complete")
    return comparison_df


def compare_prediction_head_architectures(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare models with different prediction head architectures.
    
    Args:
        models: Dictionary mapping architecture name to trained model
                e.g., {'single_layer': model_single, 'two_layer_mlp': model_mlp}
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        output_dir: Optional directory to save comparison table
    
    Returns:
        comparison_df: DataFrame with metrics for each architecture
    """
    logger.info("Comparing prediction head architectures")
    
    results = []
    
    for arch_name, model in models.items():
        logger.info(f"Evaluating model with {arch_name} architecture")
        
        # Compute all metrics
        metrics = evaluate_model(model, test_loader, device, output_dir=None)
        
        # Add architecture name
        metrics['architecture'] = arch_name
        results.append(metrics)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns to put architecture first
    cols = ['architecture'] + [col for col in comparison_df.columns if col != 'architecture']
    comparison_df = comparison_df[cols]
    
    # Save to CSV if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'architecture_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Saved architecture comparison to {output_path}")
    
    logger.info("Prediction head architecture comparison complete")
    return comparison_df


def analyze_per_class_performance(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze model performance for each CIFAR-10 class.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        class_names: List of 10 CIFAR-10 class names
        device: 'cuda' or 'cpu'
        output_dir: Optional directory to save per-class analysis
    
    Returns:
        per_class_df: DataFrame with metrics for each class
    """
    logger.info("Analyzing per-class performance")
    
    model.eval()
    model = model.to(device)
    
    # Initialize storage for each class
    class_metrics = {i: {
        'kl': [],
        'js': [],
        'true_entropy': [],
        'pred_entropy': []
    } for i in range(10)}
    
    epsilon = 1e-7
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (image, soft_label, hard_label, entropy)
            images, soft_labels, hard_labels, true_entropy = batch
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            
            # Get predictions
            pred_probs = model(images)
            pred_entropy = compute_entropy(pred_probs)
            
            # Compute KL and JS for each sample
            pred_probs_safe = pred_probs + epsilon
            soft_labels_safe = soft_labels + epsilon
            pred_probs_safe = pred_probs_safe / pred_probs_safe.sum(dim=1, keepdim=True)
            soft_labels_safe = soft_labels_safe / soft_labels_safe.sum(dim=1, keepdim=True)
            
            kl_per_sample = (soft_labels_safe * torch.log(soft_labels_safe / pred_probs_safe)).sum(dim=1)
            
            m = 0.5 * (pred_probs_safe + soft_labels_safe)
            kl_target_m = (soft_labels_safe * torch.log(soft_labels_safe / m)).sum(dim=1)
            kl_pred_m = (pred_probs_safe * torch.log(pred_probs_safe / m)).sum(dim=1)
            js_per_sample = 0.5 * kl_target_m + 0.5 * kl_pred_m
            
            # Group by class
            for i in range(len(hard_labels)):
                class_idx = hard_labels[i].item()
                class_metrics[class_idx]['kl'].append(kl_per_sample[i].item())
                class_metrics[class_idx]['js'].append(js_per_sample[i].item())
                class_metrics[class_idx]['true_entropy'].append(true_entropy[i].item())
                class_metrics[class_idx]['pred_entropy'].append(pred_entropy[i].item())
    
    # Compute statistics for each class
    results = []
    for class_idx in range(10):
        metrics = class_metrics[class_idx]
        
        # Compute Pearson correlation for this class
        if len(metrics['true_entropy']) > 1:
            pearson_r, _ = pearsonr(metrics['true_entropy'], metrics['pred_entropy'])
        else:
            pearson_r = 0.0
        
        results.append({
            'class': class_names[class_idx],
            'mean_kl': float(np.mean(metrics['kl'])),
            'std_kl': float(np.std(metrics['kl'])),
            'mean_js': float(np.mean(metrics['js'])),
            'std_js': float(np.std(metrics['js'])),
            'pearson_r': float(pearson_r),
            'mean_true_entropy': float(np.mean(metrics['true_entropy'])),
            'num_samples': len(metrics['kl'])
        })
    
    # Create DataFrame
    per_class_df = pd.DataFrame(results)
    
    # Identify best and worst classes
    best_kl_class = per_class_df.loc[per_class_df['mean_kl'].idxmin(), 'class']
    worst_kl_class = per_class_df.loc[per_class_df['mean_kl'].idxmax(), 'class']
    
    logger.info(f"Best performing class (lowest KL): {best_kl_class}")
    logger.info(f"Worst performing class (highest KL): {worst_kl_class}")
    
    # Save to CSV if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'per_class_performance.csv')
        per_class_df.to_csv(output_path, index=False)
        logger.info(f"Saved per-class performance to {output_path}")
    
    logger.info("Per-class performance analysis complete")
    return per_class_df


def identify_failure_cases(
    model: nn.Module,
    test_loader: DataLoader,
    num_cases: int = 10,
    device: str = 'cuda'
) -> List[Dict]:
    """
    Identify images with highest KL divergence (worst predictions).
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        num_cases: Number of failure cases to identify
        device: 'cuda' or 'cpu'
    
    Returns:
        failure_cases: List of dicts with failure information, sorted by KL divergence (worst first)
    """
    logger.info(f"Identifying top {num_cases} failure cases")
    
    model.eval()
    model = model.to(device)
    
    all_data = []
    epsilon = 1e-7
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (image, soft_label, hard_label, entropy)
            images, soft_labels, hard_labels, true_entropy = batch
            images_device = images.to(device)
            soft_labels_device = soft_labels.to(device)
            
            # Get predictions
            pred_probs = model(images_device)
            pred_entropy = compute_entropy(pred_probs)
            
            # Compute KL divergence for each sample
            pred_probs_safe = pred_probs + epsilon
            soft_labels_safe = soft_labels_device + epsilon
            pred_probs_safe = pred_probs_safe / pred_probs_safe.sum(dim=1, keepdim=True)
            soft_labels_safe = soft_labels_safe / soft_labels_safe.sum(dim=1, keepdim=True)
            
            kl_per_sample = (soft_labels_safe * torch.log(soft_labels_safe / pred_probs_safe)).sum(dim=1)
            
            # Store data for each sample
            for i in range(len(images)):
                all_data.append({
                    'image': images[i].cpu(),
                    'true_dist': soft_labels[i].cpu(),
                    'pred_dist': pred_probs[i].cpu(),
                    'hard_label': hard_labels[i].item(),
                    'true_entropy': true_entropy[i].item(),
                    'pred_entropy': pred_entropy[i].item(),
                    'kl_divergence': kl_per_sample[i].item()
                })
    
    # Sort by KL divergence (worst first)
    all_data.sort(key=lambda x: x['kl_divergence'], reverse=True)
    
    # Select top failures
    failure_cases = all_data[:num_cases]
    
    logger.info(f"Identified {len(failure_cases)} failure cases with KL divergence range: "
                f"[{failure_cases[-1]['kl_divergence']:.4f}, {failure_cases[0]['kl_divergence']:.4f}]")
    
    return failure_cases


def run_ablation_study():
    """Run ablation studies comparing different configurations."""
    logger.info("Running ablation studies")
    raise NotImplementedError("This is a convenience wrapper - use specific comparison functions instead")


# ============================================================================
# Robustness Testing Functions (Task 10)
# ============================================================================

def add_gaussian_noise(image: torch.Tensor, severity: int) -> torch.Tensor:
    """
    Add Gaussian noise to image.
    
    Args:
        image: Tensor of shape (C, H, W) in range [0, 1]
        severity: 1 (mild), 3 (moderate), 5 (severe)
    
    Returns:
        corrupted: Noisy image clipped to [0, 1]
    """
    noise_levels = {1: 0.04, 3: 0.12, 5: 0.20}
    std = noise_levels[severity]
    noise = torch.randn_like(image) * std
    corrupted = image + noise
    return torch.clamp(corrupted, 0, 1)


def apply_gaussian_blur(image: torch.Tensor, severity: int) -> torch.Tensor:
    """
    Apply Gaussian blur to image.
    
    Args:
        image: Tensor of shape (C, H, W)
        severity: 1 (mild), 3 (moderate), 5 (severe)
    
    Returns:
        blurred: Blurred image
    """
    from torchvision import transforms
    
    kernel_sizes = {1: 3, 3: 5, 5: 7}
    sigmas = {1: 0.5, 3: 1.0, 5: 2.0}
    
    kernel_size = kernel_sizes[severity]
    sigma = sigmas[severity]
    
    blur = transforms.GaussianBlur(kernel_size, sigma)
    return blur(image)


def reduce_contrast(image: torch.Tensor, severity: int) -> torch.Tensor:
    """
    Reduce image contrast.
    
    Args:
        image: Tensor of shape (C, H, W) in range [0, 1]
        severity: 1 (mild), 3 (moderate), 5 (severe)
    
    Returns:
        low_contrast: Image with reduced contrast clipped to [0, 1]
    """
    contrast_factors = {1: 0.8, 3: 0.5, 5: 0.3}
    factor = contrast_factors[severity]
    
    mean = image.mean(dim=(1, 2), keepdim=True)
    low_contrast = mean + factor * (image - mean)
    return torch.clamp(low_contrast, 0, 1)


def evaluate_corruption_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate model robustness to image corruptions.
    
    Tests three corruption types (Gaussian noise, Gaussian blur, contrast reduction)
    at three severity levels (1, 3, 5) and measures entropy change.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        output_dir: Optional directory to save results
    
    Returns:
        results: Dict mapping corruption_type to dict of {severity: entropy_change}
    """
    logger.info("Evaluating corruption robustness")
    
    model.eval()
    model = model.to(device)
    
    corruption_fns = {
        'gaussian_noise': add_gaussian_noise,
        'gaussian_blur': apply_gaussian_blur,
        'contrast_reduction': reduce_contrast
    }
    
    severities = [1, 3, 5]
    
    # Initialize results storage
    results = {corruption_name: {severity: [] for severity in severities} 
               for corruption_name in corruption_fns.keys()}
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (image, soft_label, hard_label, entropy)
            images, _, _, _ = batch
            images = images.to(device)
            
            # Get predictions on clean images
            clean_probs = model(images)
            clean_entropy = compute_entropy(clean_probs)
            
            # Test each corruption type and severity
            for corruption_name, corruption_fn in corruption_fns.items():
                for severity in severities:
                    # Apply corruption to each image in batch
                    corrupted = torch.stack([
                        corruption_fn(img, severity) for img in images
                    ])
                    corrupted = corrupted.to(device)
                    
                    # Get predictions on corrupted images
                    corrupted_probs = model(corrupted)
                    corrupted_entropy = compute_entropy(corrupted_probs)
                    
                    # Compute absolute entropy change
                    entropy_change = (corrupted_entropy - clean_entropy).abs()
                    results[corruption_name][severity].extend(entropy_change.cpu().numpy())
    
    # Compute mean entropy change for each corruption type and severity
    summary_results = {}
    for corruption_name in corruption_fns.keys():
        summary_results[corruption_name] = {}
        for severity in severities:
            mean_change = float(np.mean(results[corruption_name][severity]))
            summary_results[corruption_name][severity] = mean_change
            logger.info(f"{corruption_name} (severity {severity}): "
                       f"mean entropy change = {mean_change:.4f} bits")
    
    # Save results if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'corruption_robustness.json')
        with open(output_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        logger.info(f"Saved corruption robustness results to {output_path}")
    
    logger.info("Corruption robustness evaluation complete")
    return summary_results
