"""
Simple test for Phase 6: Explainability features

Tests core functionality with synthetic data (no dataset download required).
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.model import DisagreementPredictor
from src.visualization import GradCAM

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']


def test_gradcam_class():
    """Test GradCAM class implementation (Task 11.1)"""
    print("\n" + "="*70)
    print("Testing GradCAM Class")
    print("="*70)
    
    # Create model
    device = 'cpu'
    model = DisagreementPredictor()
    
    # Try to load trained checkpoint
    try:
        checkpoint = torch.load('checkpoints/finetuned_kl_demo.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained model checkpoint")
    except FileNotFoundError:
        print("⚠ Using random weights for demo")
    
    model = model.to(device)
    model.eval()
    
    # Create synthetic test image (3, 32, 32)
    test_image = torch.randn(1, 3, 32, 32)
    
    # Initialize GradCAM
    print("Initializing GradCAM...")
    gradcam = GradCAM(model, model.backbone.layer4[-1])
    print("✓ GradCAM initialized successfully")
    
    # Generate CAM
    print("Generating Grad-CAM heatmap...")
    cam = gradcam.generate_cam(test_image)
    print(f"✓ Generated heatmap with shape: {cam.shape}")
    print(f"  Heatmap value range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    # Verify heatmap properties
    assert cam.shape == (32, 32), f"Expected shape (32, 32), got {cam.shape}"
    assert cam.min() >= 0 and cam.max() <= 1, f"Expected values in [0, 1], got [{cam.min()}, {cam.max()}]"
    print("✓ Heatmap properties verified")
    
    # Clean up
    gradcam.remove_hooks()
    print("✓ Hooks removed")
    
    print("="*70)
    return True


def test_gradcam_with_different_targets():
    """Test GradCAM with different target classes"""
    print("\n" + "="*70)
    print("Testing GradCAM with Different Target Classes")
    print("="*70)
    
    # Create model
    device = 'cpu'
    model = DisagreementPredictor()
    
    try:
        checkpoint = torch.load('checkpoints/finetuned_kl_demo.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained model checkpoint")
    except FileNotFoundError:
        print("⚠ Using random weights for demo")
    
    model = model.to(device)
    model.eval()
    
    # Create synthetic test image
    test_image = torch.randn(1, 3, 32, 32)
    
    # Initialize GradCAM
    gradcam = GradCAM(model, model.backbone.layer4[-1])
    
    # Test with different target classes
    print("Testing with different target classes...")
    for target_class in [0, 5, 9]:
        cam = gradcam.generate_cam(test_image, target_class=target_class)
        print(f"  Class {target_class} ({CLASS_NAMES[target_class]}): "
              f"heatmap range [{cam.min():.3f}, {cam.max():.3f}]")
    
    print("✓ Successfully generated heatmaps for different classes")
    
    # Clean up
    gradcam.remove_hooks()
    
    print("="*70)
    return True


def test_identify_failure_cases():
    """Test failure case identification (Task 11.3)"""
    print("\n" + "="*70)
    print("Testing Failure Case Identification")
    print("="*70)
    
    from src.evaluation import identify_failure_cases
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create model
    device = 'cpu'
    model = DisagreementPredictor()
    
    try:
        checkpoint = torch.load('checkpoints/finetuned_kl_demo.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained model checkpoint")
    except FileNotFoundError:
        print("⚠ Using random weights for demo")
    
    model = model.to(device)
    model.eval()
    
    # Create synthetic test data
    num_samples = 50
    images = torch.randn(num_samples, 3, 32, 32)
    soft_labels = torch.softmax(torch.randn(num_samples, 10), dim=1)
    hard_labels = torch.randint(0, 10, (num_samples,))
    entropies = torch.rand(num_samples) * 3.0  # 0-3 bits
    
    # Create dataset and loader
    dataset = TensorDataset(images, soft_labels, hard_labels, entropies)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Identify failure cases
    print("Identifying top 10 failure cases...")
    failure_cases = identify_failure_cases(model, loader, num_cases=10, device=device)
    
    print(f"✓ Identified {len(failure_cases)} failure cases")
    print(f"  KL divergence range: [{failure_cases[-1]['kl_divergence']:.4f}, {failure_cases[0]['kl_divergence']:.4f}]")
    
    # Verify structure
    assert len(failure_cases) == 10, f"Expected 10 cases, got {len(failure_cases)}"
    assert 'image' in failure_cases[0], "Missing 'image' key"
    assert 'true_dist' in failure_cases[0], "Missing 'true_dist' key"
    assert 'pred_dist' in failure_cases[0], "Missing 'pred_dist' key"
    assert 'kl_divergence' in failure_cases[0], "Missing 'kl_divergence' key"
    print("✓ Failure case structure verified")
    
    print("="*70)
    return True


def test_categorization_summary():
    """Test categorization summary generation (Task 11.4)"""
    print("\n" + "="*70)
    print("Testing Categorization Summary")
    print("="*70)
    
    from src.visualization import generate_categorization_summary
    
    # Create mock categorization data
    categorization = {
        0: 'ambiguous_identity',
        1: 'ambiguous_identity',
        2: 'poor_image_quality',
        3: 'multi_object_scene',
        4: 'ambiguous_identity',
        5: 'boundary_case',
        6: 'poor_image_quality',
        7: 'other',
        8: 'ambiguous_identity',
        9: 'multi_object_scene'
    }
    
    print("Generating categorization summary...")
    summary = generate_categorization_summary(categorization)
    
    print(f"✓ Generated summary for {summary['total_images']} images")
    print(f"  Number of categories: {len(summary['categories'])}")
    
    # Verify structure
    assert summary['total_images'] == 10, f"Expected 10 images, got {summary['total_images']}"
    assert 'ambiguous_identity' in summary['categories'], "Missing 'ambiguous_identity' category"
    assert summary['categories']['ambiguous_identity']['count'] == 4, "Incorrect count for ambiguous_identity"
    print("✓ Summary structure verified")
    
    print("="*70)
    return True


if __name__ == '__main__':
    import os
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 6: EXPLAINABILITY SIMPLE TESTS")
    print("="*70)
    
    # Run tests
    results = []
    
    try:
        results.append(("GradCAM Class", test_gradcam_class()))
    except Exception as e:
        print(f"✗ GradCAM Class test failed: {e}")
        results.append(("GradCAM Class", False))
    
    try:
        results.append(("GradCAM Different Targets", test_gradcam_with_different_targets()))
    except Exception as e:
        print(f"✗ GradCAM Different Targets test failed: {e}")
        results.append(("GradCAM Different Targets", False))
    
    try:
        results.append(("Failure Case Identification", test_identify_failure_cases()))
    except Exception as e:
        print(f"✗ Failure Case Identification test failed: {e}")
        results.append(("Failure Case Identification", False))
    
    try:
        results.append(("Categorization Summary", test_categorization_summary()))
    except Exception as e:
        print(f"✗ Categorization Summary test failed: {e}")
        results.append(("Categorization Summary", False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<40} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("="*70 + "\n")
