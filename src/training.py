"""
Training Module

Implements two-stage training protocol: pretraining on hard labels followed by
fine-tuning on soft labels with early stopping and checkpoint management.
"""

import logging
import random
import json
import os
from typing import Dict, Callable, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .losses import kl_divergence_loss
from .data_pipeline import ConfigParseError, NumericalInstabilityError, CheckpointLoadError

logger = logging.getLogger(__name__)


def check_numerical_stability(loss: torch.Tensor, epoch: int, batch_idx: Optional[int] = None) -> None:
    """
    Check for NaN or Inf values during training.
    
    Args:
        loss: Loss tensor to check
        epoch: Current epoch number
        batch_idx: Optional batch index for more detailed error messages
    
    Raises:
        NumericalInstabilityError: If NaN or Inf detected
    """
    location = f"epoch {epoch}"
    if batch_idx is not None:
        location += f", batch {batch_idx}"
    
    if torch.isnan(loss):
        raise NumericalInstabilityError(
            f"NaN loss detected at {location}. "
            f"Try reducing learning rate or checking loss function implementation."
        )
    
    if torch.isinf(loss):
        raise NumericalInstabilityError(
            f"Inf loss detected at {location}. "
            f"Check for division by zero or overflow in loss computation."
        )


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator
    - CUDA operations (if available)
    - Makes cuDNN deterministic
    
    Args:
        seed: Random seed value (default: 42)
    """
    logger.info(f"Setting random seed to {seed}")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed (for all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed} for all random number generators")


def get_train_transform():
    """
    Get training data augmentation transform.
    
    Applies:
    - RandomHorizontalFlip with p=0.5
    - RandomCrop(32, padding=4)
    - Normalize with CIFAR-10 statistics
    
    Returns:
        transform: Composed transform for training
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2470, 0.2435, 0.2616])
    ])


def get_test_transform():
    """
    Get test/validation transform (no augmentation).
    
    Applies:
    - Normalize with CIFAR-10 statistics only
    
    Returns:
        transform: Composed transform for testing/validation
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2470, 0.2435, 0.2616])
    ])


def pretrain_on_hard_labels(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 100,
    device: str = 'cuda',
    save_path: str = 'checkpoints/pretrained_resnet18_cifar10.pth'
) -> tuple:
    """
    Pretrain model on CIFAR-10 hard labels.
    
    Uses:
    - Cross-entropy loss
    - AdamW optimizer with lr=1e-3, weight_decay=1e-4
    - Batch size 128 (configured in DataLoader)
    - Cosine annealing learning rate schedule
    
    Args:
        model: DisagreementPredictor model
        train_loader: DataLoader for CIFAR-10 training set
        num_epochs: Number of training epochs (default: 100)
        device: 'cuda' or 'cpu'
        save_path: Path to save pretrained weights
    
    Returns:
        model: Pretrained model
        history: Training history dict with 'train_loss' and 'train_acc'
    """
    logger.info(f"Starting pretraining on hard labels for {num_epochs} epochs")
    
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Check for numerical instability
            try:
                check_numerical_stability(loss, epoch, batch_idx)
            except NumericalInstabilityError as e:
                logger.critical(str(e))
                raise
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Log batch-level details at DEBUG level
            if batch_idx % 100 == 0:
                logger.debug(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Log metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(accuracy)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    # Save pretrained weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved pretrained model to {save_path}")
    
    return model, history


def finetune_on_soft_labels(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    loss_name: str = 'kl',
    num_epochs: int = 50,
    device: str = 'cuda',
    save_path: Optional[str] = None
) -> tuple:
    """
    Fine-tune pretrained model on CIFAR-10H soft labels.
    
    Uses:
    - AdamW optimizer with lr=1e-4, weight_decay=1e-4
    - Batch size 64 (configured in DataLoader)
    - Early stopping with patience=10 based on validation KL divergence
    - Saves best model checkpoint
    
    Args:
        model: Pretrained DisagreementPredictor model
        train_loader: DataLoader for CIFAR-10H training split
        val_loader: DataLoader for CIFAR-10H validation split
        loss_fn: Loss function (kl_divergence_loss, js_divergence_loss, or custom)
        loss_name: Name of loss function for checkpoint naming (default: 'kl')
        num_epochs: Maximum number of training epochs (default: 50)
        device: 'cuda' or 'cpu'
        save_path: Path to save best checkpoint (default: checkpoints/finetuned_{loss_name}_best.pth)
    
    Returns:
        model: Fine-tuned model
        history: Training history dict with 'train_loss', 'val_loss', 'val_kl', 'val_js'
    """
    from .losses import js_divergence_loss
    
    logger.info(f"Starting fine-tuning on soft labels with {loss_name} loss for up to {num_epochs} epochs")
    
    if save_path is None:
        save_path = f'checkpoints/finetuned_{loss_name}_best.pth'
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_kl': [],
        'val_js': []
    }
    
    best_val_kl = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, soft_labels, _, _) in enumerate(train_loader):
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            
            # Forward pass
            pred_probs = model(images)
            loss = loss_fn(pred_probs, soft_labels)
            
            # Check for numerical instability
            try:
                check_numerical_stability(loss, epoch, batch_idx)
            except NumericalInstabilityError as e:
                logger.critical(str(e))
                raise
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log batch-level details at DEBUG level
            if batch_idx % 50 == 0:
                logger.debug(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_kl = 0.0
        val_js = 0.0
        
        with torch.no_grad():
            for images, soft_labels, _, _ in val_loader:
                images = images.to(device)
                soft_labels = soft_labels.to(device)
                
                pred_probs = model(images)
                
                # Compute all metrics
                val_loss += loss_fn(pred_probs, soft_labels).item()
                val_kl += kl_divergence_loss(pred_probs, soft_labels).item()
                val_js += js_divergence_loss(pred_probs, soft_labels).item()
        
        # Average metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_kl /= len(val_loader)
        val_js /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_kl'].append(val_kl)
        history['val_js'].append(val_js)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                   f"Val KL: {val_kl:.4f}, Val JS: {val_js:.4f}")
        
        # Early stopping based on validation KL divergence
        if val_kl < best_val_kl:
            best_val_kl = val_kl
            patience_counter = 0
            # Save best model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path} (Val KL: {val_kl:.4f})")
        else:
            patience_counter += 1
            logger.debug(f"No improvement in validation KL. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    logger.info(f"Loaded best model from {save_path}")
    
    return model, history


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    filepath: str,
    config: Optional[Dict] = None
):
    """
    Save model checkpoint with all training state.
    
    Saves:
    - Model state dict
    - Optimizer state dict
    - Current epoch
    - Training metrics
    - Configuration (optional)
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics dict
        filepath: Path to save checkpoint
        config: Optional configuration dict
    """
    logger.info(f"Saving checkpoint to {filepath}")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved successfully")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> tuple:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics from checkpoint
        config: Configuration from checkpoint (if available)
    
    Raises:
        CheckpointLoadError: If checkpoint cannot be loaded
    """
    logger.info(f"Loading checkpoint from {filepath}")
    
    if not os.path.exists(filepath):
        raise CheckpointLoadError(f"Checkpoint file not found: {filepath}")
    
    try:
        checkpoint = torch.load(filepath)
    except Exception as e:
        raise CheckpointLoadError(f"Failed to load checkpoint from {filepath}: {str(e)}")
    
    try:
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        config = checkpoint.get('config', None)
        
        logger.info(f"Checkpoint loaded successfully (epoch {epoch})")
        
        return epoch, metrics, config
    except Exception as e:
        raise CheckpointLoadError(f"Failed to load checkpoint state from {filepath}: {str(e)}")


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters.
    
    Attributes:
        pretrain_epochs: Number of pretraining epochs (default: 100)
        finetune_epochs: Number of fine-tuning epochs (default: 50)
        pretrain_lr: Learning rate for pretraining (default: 1e-3)
        finetune_lr: Learning rate for fine-tuning (default: 1e-4)
        weight_decay: Weight decay for AdamW (default: 1e-4)
        pretrain_batch_size: Batch size for pretraining (default: 128)
        finetune_batch_size: Batch size for fine-tuning (default: 64)
        early_stopping_patience: Patience for early stopping (default: 10)
        random_seed: Random seed (default: 42)
        loss_function: Loss function name ('kl', 'js', or 'custom') (default: 'kl')
        lambda_weight: Weight for entropy penalty in custom loss (default: 0.1)
    """
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    pretrain_batch_size: int = 128
    finetune_batch_size: int = 64
    early_stopping_patience: int = 10
    random_seed: int = 42
    loss_function: str = 'kl'
    lambda_weight: float = 0.1
    
    def to_json(self, filepath: str):
        """
        Serialize configuration to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Saved training configuration to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> 'TrainingConfig':
        """
        Deserialize configuration from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            TrainingConfig instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            config = cls(**data)
            config.validate()
            
            logger.info(f"Loaded training configuration from {filepath}")
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigParseError(f"Invalid JSON in configuration file: {e}")
        except TypeError as e:
            raise ConfigParseError(f"Invalid configuration fields: {e}")
    
    def validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.pretrain_epochs <= 0:
            raise ValueError(f"pretrain_epochs must be positive, got {self.pretrain_epochs}")
        
        if self.finetune_epochs <= 0:
            raise ValueError(f"finetune_epochs must be positive, got {self.finetune_epochs}")
        
        if self.pretrain_lr <= 0:
            raise ValueError(f"pretrain_lr must be positive, got {self.pretrain_lr}")
        
        if self.finetune_lr <= 0:
            raise ValueError(f"finetune_lr must be positive, got {self.finetune_lr}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if self.pretrain_batch_size <= 0:
            raise ValueError(f"pretrain_batch_size must be positive, got {self.pretrain_batch_size}")
        
        if self.finetune_batch_size <= 0:
            raise ValueError(f"finetune_batch_size must be positive, got {self.finetune_batch_size}")
        
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.early_stopping_patience}")
        
        if self.random_seed < 0:
            raise ValueError(f"random_seed must be non-negative, got {self.random_seed}")
        
        valid_loss_functions = ['kl', 'js', 'custom']
        if self.loss_function not in valid_loss_functions:
            raise ValueError(
                f"loss_function must be one of {valid_loss_functions}, got '{self.loss_function}'"
            )
        
        if self.lambda_weight < 0:
            raise ValueError(f"lambda_weight must be non-negative, got {self.lambda_weight}")
    
    @staticmethod
    def get_json_schema() -> dict:
        """
        Get JSON schema for validation.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "pretrain_epochs": {"type": "integer", "minimum": 1},
                "finetune_epochs": {"type": "integer", "minimum": 1},
                "pretrain_lr": {"type": "number", "exclusiveMinimum": 0},
                "finetune_lr": {"type": "number", "exclusiveMinimum": 0},
                "weight_decay": {"type": "number", "minimum": 0},
                "pretrain_batch_size": {"type": "integer", "minimum": 1},
                "finetune_batch_size": {"type": "integer", "minimum": 1},
                "early_stopping_patience": {"type": "integer", "minimum": 1},
                "random_seed": {"type": "integer", "minimum": 0},
                "loss_function": {"type": "string", "enum": ["kl", "js", "custom"]},
                "lambda_weight": {"type": "number", "minimum": 0}
            },
            "required": [
                "pretrain_epochs", "finetune_epochs", "pretrain_lr", "finetune_lr",
                "weight_decay", "pretrain_batch_size", "finetune_batch_size",
                "early_stopping_patience", "random_seed", "loss_function", "lambda_weight"
            ]
        }



def train_all_models(
    pretrained_model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    num_epochs: int = 50
) -> Dict[str, tuple]:
    """
    Train three models with different loss functions.
    
    Trains models with:
    - KL divergence loss
    - JS divergence loss
    - Custom entropy-regularized loss
    
    Args:
        pretrained_model_path: Path to pretrained model weights
        train_loader: DataLoader for CIFAR-10H training split
        val_loader: DataLoader for CIFAR-10H validation split
        device: 'cuda' or 'cpu'
        num_epochs: Maximum number of training epochs
    
    Returns:
        results: Dict mapping loss name to (model, history) tuple
    """
    from .model import DisagreementPredictor
    from .losses import kl_divergence_loss, js_divergence_loss, custom_entropy_regularized_loss
    
    logger.info("Training three models with different loss functions")
    
    loss_functions = {
        'kl': kl_divergence_loss,
        'js': js_divergence_loss,
        'custom': custom_entropy_regularized_loss
    }
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model with {loss_name.upper()} loss")
        logger.info(f"{'='*60}\n")
        
        # Create new model and load pretrained weights
        model = DisagreementPredictor()
        model.load_state_dict(torch.load(pretrained_model_path))
        
        # Fine-tune on soft labels
        model, history = finetune_on_soft_labels(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            loss_name=loss_name,
            num_epochs=num_epochs,
            device=device
        )
        
        results[loss_name] = (model, history)
        
        logger.info(f"Completed training with {loss_name.upper()} loss")
        logger.info(f"Best Val KL: {min(history['val_kl']):.4f}")
        logger.info(f"Best Val JS: {min(history['val_js']):.4f}\n")
    
    logger.info("All three models trained successfully")
    return results
