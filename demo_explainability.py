"""
Demo script for Phase 6: Explainability features

Tests Grad-CAM visualization, failure case analysis, and manual categorization interface.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.model import DisagreementPredictor
from src.data_pipeline import (
    load_cifar10_data,
    load_cifar10h_data,
    compute_soft_labels,
    align_datasets,
    split_dataset,
    CIFAR10HDataset
)
from src.visualization import (
    visualize_gradcam_comparison,
    visualize_failure_cases,
    manual_categorization_interface,
    generate_categorization_summary
)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_test_data():
    """Load CIFAR-10H test dataset"""
    # Load CIFAR-10 test data (don't download, use existing)
    cifar10_images, cifar10_labels = load_cifar10_data(train=False, download=False)
    
    # Load CIFAR-10H data
    cifar10h_counts, cifar10h_probs = load_cifar10h_data()
    
    # Compute soft labels
    cifar10h_soft_labels = compute_soft_labels(cifar10h_counts)
    
    # Align datasets
    aligned_data = align_datasets(cifar10_images, cifar10_labels, cifar10h_soft_labels)
    
    # Split dataset
    _, _, test_data = split_dataset(aligned_data, random_seed=42)
    
    # Create dataset
    test_dataset = CIFAR10HDataset(test_data)
    
    return test_dataset


def test_gradcam_visualization():
    """Test Grad-CAM visualization (Task 11.1 & 11.2)"""
    print("\n" + "="*70)
    print("Testing Grad-CAM Visualization")
    print("="*70)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = DisagreementPredictor()
    
    # Try to load a trained checkpoint
    try:
        checkpoint = torch.load('checkpoints/finetuned_kl_demo.pth', map_location=device)
        # Check if checkpoint is a dict with 'model_state_dict' key or directly the state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained model checkpoint")
    except FileNotFoundError:
        print("⚠ No trained checkpoint found, using random weights for demo")
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = load_test_data()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Collect images and entropies
    all_images = []
    all_entropies = []
    
    with torch.no_grad():
        for images, _, _, entropies in test_loader:
            all_images.append(images)
            all_entropies.append(entropies)
    
    all_images = torch.cat(all_images)
    all_entropies = torch.cat(all_entropies)
    
    # Select low and high entropy images
    low_indices = torch.argsort(all_entropies)[:5]
    high_indices = torch.argsort(all_entropies)[-5:]
    
    low_entropy_images = all_images[low_indices]
    high_entropy_images = all_images[high_indices]
    
    print(f"Selected 5 low-entropy images (entropy range: {all_entropies[low_indices].min():.3f} - {all_entropies[low_indices].max():.3f})")
    print(f"Selected 5 high-entropy images (entropy range: {all_entropies[high_indices].min():.3f} - {all_entropies[high_indices].max():.3f})")
    
    # Generate Grad-CAM comparison visualization
    visualize_gradcam_comparison(
        model=model,
        low_entropy_images=low_entropy_images,
        high_entropy_images=high_entropy_images,
        save_path='outputs/gradcam_comparison.png',
        device=device
    )
    
    print("✓ Grad-CAM comparison visualization saved to outputs/gradcam_comparison.png")
    print("="*70)


def test_failure_case_analysis():
    """Test failure case analysis (Task 11.3)"""
    print("\n" + "="*70)
    print("Testing Failure Case Analysis")
    print("="*70)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = DisagreementPredictor()
    
    # Try to load a trained checkpoint
    try:
        checkpoint = torch.load('checkpoints/finetuned_kl_demo.pth', map_location=device)
        # Check if checkpoint is a dict with 'model_state_dict' key or directly the state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained model checkpoint")
    except FileNotFoundError:
        print("⚠ No trained checkpoint found, using random weights for demo")
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = load_test_data()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Visualize top 10 failure cases
    visualize_failure_cases(
        model=model,
        test_loader=test_loader,
        num_cases=10,
        save_path='outputs/failure_cases.png',
        device=device,
        class_names=CLASS_NAMES
    )
    
    print("✓ Failure case visualization saved to outputs/failure_cases.png")
    print("="*70)


def test_manual_categorization_interface():
    """Test manual categorization interface (Task 11.4)"""
    print("\n" + "="*70)
    print("Testing Manual Categorization Interface")
    print("="*70)
    print("\nNOTE: This is an interactive feature that requires user input.")
    print("Skipping in automated demo. To test manually, run:")
    print("  python -c \"from demo_explainability import run_manual_categorization; run_manual_categorization()\"")
    print("="*70)


def run_manual_categorization():
    """Run manual categorization interface (for manual testing)"""
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DisagreementPredictor()
    
    # Try to load a trained checkpoint
    try:
        checkpoint = torch.load('checkpoints/finetuned_kl_demo.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained model checkpoint")
    except FileNotFoundError:
        print("⚠ No trained checkpoint found, using random weights for demo")
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = load_test_data()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Run manual categorization (interactive)
    categorization = manual_categorization_interface(
        model=model,
        test_loader=test_loader,
        num_images=25,
        device=device,
        class_names=CLASS_NAMES
    )
    
    # Generate summary report
    summary = generate_categorization_summary(
        categorization=categorization,
        save_path='outputs/categorization_summary.json'
    )
    
    print("✓ Categorization summary saved to outputs/categorization_summary.json")


if __name__ == '__main__':
    import os
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 6: EXPLAINABILITY DEMO")
    print("="*70)
    
    # Test each component
    test_gradcam_visualization()
    test_failure_case_analysis()
    test_manual_categorization_interface()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - outputs/gradcam_comparison.png")
    print("  - outputs/failure_cases.png")
    print("\nTo test manual categorization interface, run:")
    print("  python -c \"from demo_explainability import run_manual_categorization; run_manual_categorization()\"")
    print("="*70 + "\n")
