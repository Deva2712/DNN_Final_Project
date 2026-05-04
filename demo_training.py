"""
Demonstration script for training protocol.

This script demonstrates the two-stage training protocol:
1. Pretraining on CIFAR-10 hard labels
2. Fine-tuning on CIFAR-10H soft labels with three different loss functions

Note: This is a demonstration script. For actual training, you would need to:
- Load the actual CIFAR-10 and CIFAR-10H datasets
- Use proper data loaders with the correct batch sizes
- Run for the full number of epochs (100 for pretraining, 50 for fine-tuning)
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import DisagreementPredictor
from src.training import (
    set_seed,
    get_train_transform,
    get_test_transform,
    pretrain_on_hard_labels,
    finetune_on_soft_labels,
    train_all_models,
    TrainingConfig,
    save_checkpoint,
    load_checkpoint
)
from src.losses import kl_divergence_loss, js_divergence_loss, custom_entropy_regularized_loss


def create_mock_data(num_samples=100, batch_size=32):
    """Create mock data for demonstration."""
    # Mock CIFAR-10 data (hard labels)
    cifar10_images = torch.randn(num_samples, 3, 32, 32)
    cifar10_labels = torch.randint(0, 10, (num_samples,))
    cifar10_dataset = TensorDataset(cifar10_images, cifar10_labels)
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
    
    # Mock CIFAR-10H data (soft labels)
    cifar10h_images = torch.randn(num_samples, 3, 32, 32)
    cifar10h_soft_labels = torch.softmax(torch.randn(num_samples, 10), dim=1)
    cifar10h_hard_labels = torch.randint(0, 10, (num_samples,))
    cifar10h_entropies = torch.rand(num_samples) * 3.32  # Random entropy values
    
    cifar10h_dataset = TensorDataset(
        cifar10h_images, cifar10h_soft_labels, cifar10h_hard_labels, cifar10h_entropies
    )
    cifar10h_loader = DataLoader(cifar10h_dataset, batch_size=batch_size, shuffle=True)
    
    return cifar10_loader, cifar10h_loader


def main():
    """Demonstrate the training protocol."""
    print("="*80)
    print("CIFAR-10 Human Disagreement Predictor - Training Protocol Demo")
    print("="*80)
    print()
    
    # Step 1: Set random seed for reproducibility
    print("Step 1: Setting random seed for reproducibility")
    set_seed(42)
    print("✓ Random seed set to 42")
    print()
    
    # Step 2: Create training configuration
    print("Step 2: Creating training configuration")
    config = TrainingConfig(
        pretrain_epochs=2,  # Reduced for demo
        finetune_epochs=2,  # Reduced for demo
        pretrain_lr=1e-3,
        finetune_lr=1e-4,
        weight_decay=1e-4,
        pretrain_batch_size=32,
        finetune_batch_size=32,
        early_stopping_patience=10,
        random_seed=42,
        loss_function='kl'
    )
    print(f"✓ Configuration created:")
    print(f"  - Pretrain epochs: {config.pretrain_epochs}")
    print(f"  - Finetune epochs: {config.finetune_epochs}")
    print(f"  - Pretrain LR: {config.pretrain_lr}")
    print(f"  - Finetune LR: {config.finetune_lr}")
    print()
    
    # Step 3: Save configuration
    print("Step 3: Saving configuration to JSON")
    config.to_json('outputs/training_config.json')
    print("✓ Configuration saved to outputs/training_config.json")
    print()
    
    # Step 4: Create mock data loaders
    print("Step 4: Creating mock data loaders")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    cifar10_loader, cifar10h_loader = create_mock_data(num_samples=100, batch_size=32)
    print("✓ Mock data loaders created")
    print(f"  - CIFAR-10 loader: {len(cifar10_loader)} batches")
    print(f"  - CIFAR-10H loader: {len(cifar10h_loader)} batches")
    print()
    
    # Step 5: Create model
    print("Step 5: Creating DisagreementPredictor model")
    model = DisagreementPredictor()
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Step 6: Pretrain on hard labels
    print("Step 6: Pretraining on CIFAR-10 hard labels")
    print("(Running for 2 epochs with mock data for demonstration)")
    model, pretrain_history = pretrain_on_hard_labels(
        model=model,
        train_loader=cifar10_loader,
        num_epochs=config.pretrain_epochs,
        device=device,
        save_path='checkpoints/pretrained_demo.pth'
    )
    print("✓ Pretraining completed")
    print(f"  - Final train loss: {pretrain_history['train_loss'][-1]:.4f}")
    print(f"  - Final train accuracy: {pretrain_history['train_acc'][-1]:.2f}%")
    print()
    
    # Step 7: Fine-tune with KL divergence loss
    print("Step 7: Fine-tuning on CIFAR-10H soft labels with KL divergence loss")
    print("(Running for 2 epochs with mock data for demonstration)")
    model_kl, finetune_history = finetune_on_soft_labels(
        model=model,
        train_loader=cifar10h_loader,
        val_loader=cifar10h_loader,  # Using same for demo
        loss_fn=kl_divergence_loss,
        loss_name='kl',
        num_epochs=config.finetune_epochs,
        device=device,
        save_path='checkpoints/finetuned_kl_demo.pth'
    )
    print("✓ Fine-tuning completed")
    print(f"  - Final train loss: {finetune_history['train_loss'][-1]:.4f}")
    print(f"  - Final val KL: {finetune_history['val_kl'][-1]:.4f}")
    print(f"  - Final val JS: {finetune_history['val_js'][-1]:.4f}")
    print()
    
    # Step 8: Demonstrate checkpoint management
    print("Step 8: Demonstrating checkpoint management")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = {
        'train_loss': finetune_history['train_loss'][-1],
        'val_kl': finetune_history['val_kl'][-1]
    }
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.finetune_epochs,
        metrics=metrics,
        filepath='checkpoints/demo_checkpoint.pth',
        config={'loss_function': 'kl', 'lr': 1e-4}
    )
    print("✓ Checkpoint saved")
    
    # Load checkpoint
    new_model = DisagreementPredictor()
    epoch, loaded_metrics, loaded_config = load_checkpoint(
        'checkpoints/demo_checkpoint.pth',
        new_model
    )
    print("✓ Checkpoint loaded")
    print(f"  - Epoch: {epoch}")
    print(f"  - Metrics: {loaded_metrics}")
    print()
    
    # Step 9: Show data augmentation transforms
    print("Step 9: Data augmentation transforms")
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    print("✓ Training transform (with augmentation):")
    print(f"  {train_transform}")
    print("✓ Test transform (no augmentation):")
    print(f"  {test_transform}")
    print()
    
    print("="*80)
    print("Demo completed successfully!")
    print("="*80)
    print()
    print("Summary of implemented features:")
    print("✓ Task 6.1: Random seed management (set_seed)")
    print("✓ Task 6.2: Data augmentation transforms (get_train_transform, get_test_transform)")
    print("✓ Task 6.3: Pretraining on CIFAR-10 hard labels (pretrain_on_hard_labels)")
    print("✓ Task 6.4: Fine-tuning on CIFAR-10H soft labels (finetune_on_soft_labels)")
    print("✓ Task 6.5: Support for training with different loss functions (train_all_models)")
    print("✓ Task 6.6: Checkpoint management (save_checkpoint, load_checkpoint)")
    print("✓ Task 6.7: Training configuration serialization (TrainingConfig)")
    print()
    print("Next steps:")
    print("- Load actual CIFAR-10 and CIFAR-10H datasets")
    print("- Train for full epochs (100 pretrain, 50 finetune)")
    print("- Train all three models (KL, JS, Custom loss)")
    print("- Evaluate models on test set")


if __name__ == '__main__':
    main()
