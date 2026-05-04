"""
Demo script for Phase 1: Data Pipeline

Demonstrates the complete data pipeline including:
- Loading CIFAR-10 and CIFAR-10H datasets
- Computing soft labels
- Aligning datasets
- Splitting into train/val/test
- Computing entropy
- Creating PyTorch Dataset
- Generating visualizations
- Saving/loading configuration
"""

import os
import numpy as np
import torch
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

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    print("=" * 80)
    print("CIFAR-10 Human Disagreement Predictor - Data Pipeline Demo")
    print("=" * 80)
    
    # Step 1: Load CIFAR-10H data (already available)
    print("\n[1/9] Loading CIFAR-10H dataset...")
    counts, probs = load_cifar10h_data()
    print(f"  ✓ Loaded CIFAR-10H: counts {counts.shape}, probs {probs.shape}")
    
    # Step 2: Compute soft labels from counts (verify they match probs)
    print("\n[2/9] Computing soft labels from annotator counts...")
    soft_labels = compute_soft_labels(counts)
    print(f"  ✓ Computed soft labels: {soft_labels.shape}")
    print(f"  ✓ Soft labels match provided probs: {np.allclose(soft_labels, probs, atol=1e-6)}")
    
    # Step 3: Load CIFAR-10 test set (use existing if available, otherwise create mock)
    print("\n[3/9] Loading CIFAR-10 test set...")
    try:
        cifar10_images, cifar10_labels = load_cifar10_data(train=False, download=False)
        print(f"  ✓ Loaded CIFAR-10 test set: {cifar10_images.shape}")
    except:
        print("  ⚠ CIFAR-10 not available, creating mock data for demo...")
        # Create mock CIFAR-10 test data
        cifar10_images = np.random.randint(0, 256, size=(10000, 3, 32, 32), dtype=np.uint8)
        cifar10_labels = np.random.randint(0, 10, size=10000)
        print(f"  ✓ Created mock CIFAR-10 test set: {cifar10_images.shape}")
    
    # Step 4: Align datasets
    print("\n[4/9] Aligning CIFAR-10H with CIFAR-10 test set...")
    aligned_data = align_datasets(cifar10_images, cifar10_labels, soft_labels)
    print(f"  ✓ Aligned {len(aligned_data)} image-label pairs")
    
    # Step 5: Split dataset
    print("\n[5/9] Splitting dataset (6000 train / 2000 val / 2000 test)...")
    train_data, val_data, test_data = split_dataset(aligned_data, random_seed=42)
    print(f"  ✓ Train: {len(train_data)} samples")
    print(f"  ✓ Val: {len(val_data)} samples")
    print(f"  ✓ Test: {len(test_data)} samples")
    
    # Step 6: Compute entropy for all splits
    print("\n[6/9] Computing Shannon entropy for all samples...")
    
    def compute_entropy_for_split(split_data):
        soft_labels_split = np.array([soft_label for _, soft_label, _ in split_data])
        return compute_entropy(soft_labels_split)
    
    train_entropies = compute_entropy_for_split(train_data)
    val_entropies = compute_entropy_for_split(val_data)
    test_entropies = compute_entropy_for_split(test_data)
    
    all_entropies = np.concatenate([train_entropies, val_entropies, test_entropies])
    
    print(f"  ✓ Train entropy: min={train_entropies.min():.3f}, max={train_entropies.max():.3f}, mean={train_entropies.mean():.3f}")
    print(f"  ✓ Val entropy: min={val_entropies.min():.3f}, max={val_entropies.max():.3f}, mean={val_entropies.mean():.3f}")
    print(f"  ✓ Test entropy: min={test_entropies.min():.3f}, max={test_entropies.max():.3f}, mean={test_entropies.mean():.3f}")
    
    # Step 7: Create PyTorch Dataset
    print("\n[7/9] Creating PyTorch Dataset for training...")
    
    # Convert train data to tensors
    train_images = torch.from_numpy(np.array([img for img, _, _ in train_data])).float()
    train_soft_labels = torch.from_numpy(np.array([soft_label for _, soft_label, _ in train_data])).float()
    train_hard_labels = torch.from_numpy(np.array([hard_label for _, _, hard_label in train_data])).long()
    train_entropies_tensor = torch.from_numpy(train_entropies).float()
    
    train_dataset = CIFAR10HDataset(
        train_images, train_soft_labels, train_hard_labels, train_entropies_tensor
    )
    
    print(f"  ✓ Created CIFAR10HDataset with {len(train_dataset)} samples")
    
    # Test dataset access
    sample_img, sample_soft, sample_hard, sample_entropy = train_dataset[0]
    print(f"  ✓ Sample shapes: image={sample_img.shape}, soft_label={sample_soft.shape}, hard_label={sample_hard.shape}")
    
    # Step 8: Generate visualizations
    print("\n[8/9] Generating visualizations...")
    os.makedirs('outputs/data_visualizations', exist_ok=True)
    
    # Entropy histogram
    plot_entropy_histogram(
        all_entropies,
        'outputs/data_visualizations/entropy_histogram.png'
    )
    print("  ✓ Saved entropy histogram")
    
    # Per-class entropy
    all_hard_labels = np.array([hard_label for _, _, hard_label in aligned_data])
    plot_per_class_entropy(
        all_entropies,
        all_hard_labels,
        CLASS_NAMES,
        'outputs/data_visualizations/per_class_entropy.png'
    )
    print("  ✓ Saved per-class entropy plot")
    
    # Example grid
    all_images = np.array([img for img, _, _ in aligned_data])
    all_soft_labels = np.array([soft_label for _, soft_label, _ in aligned_data])
    plot_example_grid(
        all_images,
        all_entropies,
        all_soft_labels,
        'outputs/data_visualizations/example_grid.png',
        CLASS_NAMES
    )
    print("  ✓ Saved example grid")
    
    # Step 9: Save and load configuration
    print("\n[9/9] Testing configuration serialization...")
    config = DataPipelineConfig(
        cifar10_data_dir='./data',
        cifar10h_data_dir='./cifar-10h-1.0.0/data',
        random_seed=42
    )
    
    config_path = 'outputs/data_pipeline_config.json'
    config.to_json(config_path)
    print(f"  ✓ Saved configuration to {config_path}")
    
    loaded_config = DataPipelineConfig.from_json(config_path)
    print(f"  ✓ Loaded configuration: random_seed={loaded_config.random_seed}, train_size={loaded_config.train_size}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Data Pipeline Demo Complete!")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(aligned_data)}")
    print(f"  Train/Val/Test split: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    print(f"  Entropy range: [{all_entropies.min():.3f}, {all_entropies.max():.3f}] bits")
    print(f"  Mean entropy: {all_entropies.mean():.3f} bits")
    print(f"\nOutputs saved to:")
    print(f"  - outputs/data_visualizations/entropy_histogram.png")
    print(f"  - outputs/data_visualizations/per_class_entropy.png")
    print(f"  - outputs/data_visualizations/example_grid.png")
    print(f"  - outputs/data_pipeline_config.json")
    print("\n✓ All Phase 1 tasks completed successfully!")


if __name__ == '__main__':
    main()
