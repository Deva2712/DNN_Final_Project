"""
Demo script for Phase 6: Robustness Testing

This script evaluates model robustness to image corruptions:
- Gaussian noise (severity 1, 3, 5)
- Gaussian blur (severity 1, 3, 5)
- Contrast reduction (severity 1, 3, 5)

Measures entropy change compared to clean images and generates visualization.
"""

import os
import logging
import torch
from torch.utils.data import DataLoader

from src.logging_config import setup_logging
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
from src.evaluation import evaluate_corruption_robustness
from src.visualization import plot_corruption_robustness
from torchvision import transforms

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Run robustness testing evaluation."""
    logger.info("=" * 80)
    logger.info("PHASE 6: ROBUSTNESS TESTING")
    logger.info("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = 'outputs/robustness'
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # Step 1: Load and prepare test data
    # ========================================================================
    logger.info("\nStep 1: Loading test data...")
    
    # Load CIFAR-10 test set
    cifar10_images, cifar10_labels = load_cifar10_data(
        data_dir='./data', train=False, download=True
    )
    
    # Load CIFAR-10H data
    cifar10h_counts, cifar10h_probs = load_cifar10h_data(
        data_dir='./cifar-10h-1.0.0/data'
    )
    
    # Compute soft labels
    soft_labels = compute_soft_labels(cifar10h_counts)
    
    # Align datasets
    aligned_data = align_datasets(
        cifar10_images, cifar10_labels, soft_labels
    )
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        aligned_data, random_seed=42
    )
    
    logger.info(f"Test set size: {len(test_data)}")
    
    # ========================================================================
    # Step 2: Create test dataset and dataloader
    # ========================================================================
    logger.info("\nStep 2: Creating test dataloader...")
    
    # Extract test data components
    test_images = torch.tensor(
        np.array([img for img, _, _ in test_data]), dtype=torch.float32
    ) / 255.0  # Normalize to [0, 1]
    
    test_soft_labels = torch.tensor(
        np.array([soft for _, soft, _ in test_data]), dtype=torch.float32
    )
    
    test_hard_labels = torch.tensor(
        np.array([hard for _, _, hard in test_data]), dtype=torch.long
    )
    
    test_entropies = torch.tensor(
        compute_entropy(test_soft_labels.numpy()), dtype=torch.float32
    )
    
    # Create test transform (normalization only, no augmentation)
    test_transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # Create dataset
    test_dataset = CIFAR10HDataset(
        images=test_images,
        soft_labels=test_soft_labels,
        hard_labels=test_hard_labels,
        entropies=test_entropies,
        transform=test_transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Test dataloader created with {len(test_loader)} batches")
    
    # ========================================================================
    # Step 3: Load trained model
    # ========================================================================
    logger.info("\nStep 3: Loading trained model...")
    
    # Try to load the best model (KL-trained by default)
    checkpoint_path = 'checkpoints/finetuned_kl_best.pth'
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Trying alternative checkpoint...")
        checkpoint_path = 'checkpoints/finetuned_kl_demo.pth'
    
    if not os.path.exists(checkpoint_path):
        logger.error("No trained model checkpoint found!")
        logger.error("Please train a model first using demo_training.py")
        return
    
    # Load model
    model = DisagreementPredictor()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # ========================================================================
    # Step 4: Evaluate corruption robustness
    # ========================================================================
    logger.info("\nStep 4: Evaluating corruption robustness...")
    logger.info("Testing 3 corruption types at 3 severity levels each...")
    logger.info("This may take a few minutes...")
    
    results = evaluate_corruption_robustness(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    # ========================================================================
    # Step 5: Generate visualization
    # ========================================================================
    logger.info("\nStep 5: Generating robustness visualization...")
    
    plot_path = os.path.join(output_dir, 'corruption_robustness_plot.png')
    plot_corruption_robustness(results, plot_path)
    
    logger.info(f"Saved robustness plot to {plot_path}")
    
    # ========================================================================
    # Step 6: Print summary
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


if __name__ == '__main__':
    import numpy as np
    main()
