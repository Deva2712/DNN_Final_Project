"""
Minimal demo for Phase 6: Robustness Testing

This script demonstrates the robustness testing functionality using synthetic data.
For full evaluation, use demo_robustness_testing.py with downloaded datasets.
"""

import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.logging_config import setup_logging
from src.model import DisagreementPredictor
from src.evaluation import evaluate_corruption_robustness
from src.visualization import plot_corruption_robustness

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def create_synthetic_test_data(num_samples=200):
    """Create synthetic test data for demonstration."""
    logger.info(f"Creating {num_samples} synthetic test samples...")
    
    # Create random images in [0, 1] range
    images = torch.rand(num_samples, 3, 32, 32)
    
    # Create random soft labels (probability distributions)
    soft_labels = torch.rand(num_samples, 10)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
    
    # Create random hard labels
    hard_labels = torch.randint(0, 10, (num_samples,))
    
    # Create random entropies
    entropies = torch.rand(num_samples) * 3.32  # Max entropy for 10 classes
    
    return images, soft_labels, hard_labels, entropies


def main():
    """Run minimal robustness testing demonstration."""
    logger.info("=" * 80)
    logger.info("PHASE 6: ROBUSTNESS TESTING (MINIMAL DEMO)")
    logger.info("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = 'outputs/robustness'
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # Step 1: Create synthetic test data
    # ========================================================================
    logger.info("\nStep 1: Creating synthetic test data...")
    
    images, soft_labels, hard_labels, entropies = create_synthetic_test_data(num_samples=200)
    
    # Normalize images (simulate CIFAR-10 normalization)
    from torchvision import transforms
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(images, soft_labels, hard_labels, entropies)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    logger.info(f"Created test dataloader with {len(test_loader)} batches")
    
    # ========================================================================
    # Step 2: Load or create model
    # ========================================================================
    logger.info("\nStep 2: Loading model...")
    
    # Try to load trained model, otherwise use random initialization
    checkpoint_path = 'checkpoints/finetuned_kl_demo.pth'
    
    model = DisagreementPredictor()
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading trained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Using randomly initialized model for demonstration")
    
    model = model.to(device)
    model.eval()
    
    logger.info("Model ready for evaluation")
    
    # ========================================================================
    # Step 3: Evaluate corruption robustness
    # ========================================================================
    logger.info("\nStep 3: Evaluating corruption robustness...")
    logger.info("Testing 3 corruption types at 3 severity levels each...")
    logger.info("This demonstrates the robustness evaluation functionality...")
    
    results = evaluate_corruption_robustness(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    # ========================================================================
    # Step 4: Generate visualization
    # ========================================================================
    logger.info("\nStep 4: Generating robustness visualization...")
    
    plot_path = os.path.join(output_dir, 'corruption_robustness_plot.png')
    plot_corruption_robustness(results, plot_path)
    
    logger.info(f"Saved robustness plot to {plot_path}")
    
    # ========================================================================
    # Step 5: Print summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ROBUSTNESS TESTING SUMMARY")
    logger.info("=" * 80)
    
    for corruption_type in ['gaussian_noise', 'gaussian_blur', 'contrast_reduction']:
        logger.info(f"\n{corruption_type.replace('_', ' ').title()}:")
        for severity in [1, 3, 5]:
            entropy_change = results[corruption_type][severity]
            logger.info(f"  Severity {severity}: {entropy_change:.4f} bits mean entropy change")
    
    logger.info("\n" + "=" * 80)
    logger.info("ROBUSTNESS TESTING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}/")
    logger.info(f"  - corruption_robustness.json (numerical results)")
    logger.info(f"  - corruption_robustness_plot.png (visualization)")
    logger.info("\nNOTE: This demo used synthetic data for demonstration.")
    logger.info("For real evaluation, use demo_robustness_testing.py with actual CIFAR-10H data.")


if __name__ == '__main__':
    main()
