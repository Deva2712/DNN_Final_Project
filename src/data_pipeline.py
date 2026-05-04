"""
Data Pipeline Module

Handles downloading, preprocessing, aligning, and splitting CIFAR-10 and CIFAR-10H datasets.
Computes soft labels, Shannon entropy, and provides PyTorch Dataset classes.
"""

import logging
import json
from typing import Tuple, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class CIFAR10HDataset(Dataset):
    """
    Custom PyTorch Dataset for CIFAR-10H with soft labels.
    
    Attributes:
        images: Tensor of shape (N, 3, 32, 32) - RGB images
        soft_labels: Tensor of shape (N, 10) - probability distributions
        hard_labels: Tensor of shape (N,) - original CIFAR-10 labels
        entropies: Tensor of shape (N,) - Shannon entropy values
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        images: torch.Tensor,
        soft_labels: torch.Tensor,
        hard_labels: torch.Tensor,
        entropies: torch.Tensor,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize CIFAR-10H dataset.
        
        Args:
            images: Tensor of shape (N, 3, 32, 32)
            soft_labels: Tensor of shape (N, 10)
            hard_labels: Tensor of shape (N,)
            entropies: Tensor of shape (N,)
            transform: Optional transform for data augmentation
        """
        self.images = images
        self.soft_labels = soft_labels
        self.hard_labels = hard_labels
        self.entropies = entropies
        self.transform = transform
        
        logger.info(f"Initialized CIFAR10HDataset with {len(self)} samples")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (image, soft_label, hard_label, entropy)
        """
        image = self.images[idx]
        soft_label = self.soft_labels[idx]
        hard_label = self.hard_labels[idx]
        entropy = self.entropies[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, soft_label, hard_label, entropy
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)


# Custom Exceptions
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataShapeError(Exception):
    """Raised when data has incorrect shape."""
    pass


class ConfigParseError(Exception):
    """Raised when configuration parsing fails."""
    pass


class NumericalInstabilityError(Exception):
    """Raised when NaN or Inf values are detected during training."""
    pass


class CheckpointLoadError(Exception):
    """Raised when model checkpoint cannot be loaded."""
    pass


# Data loading functions
def load_cifar10_data(data_dir: str = './data', train: bool = True, download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset using torchvision.
    
    Args:
        data_dir: Directory to store/load CIFAR-10 data
        train: If True, load training set; if False, load test set
        download: If True, download dataset if not present
    
    Returns:
        images: numpy array of shape (N, 3, 32, 32) with pixel values in [0, 255]
        labels: numpy array of shape (N,) with class labels in [0, 9]
    
    Raises:
        FileNotFoundError: If dataset not found and download=False
        DataShapeError: If loaded data has unexpected shape
    """
    logger.info(f"Loading CIFAR-10 {'training' if train else 'test'} dataset from {data_dir}...")
    
    try:
        # Load CIFAR-10 without transforms to get raw data
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=download, transform=None)
        
        # Extract images and labels
        images = np.array([np.array(img) for img, _ in dataset])  # Shape: (N, 32, 32, 3)
        labels = np.array([label for _, label in dataset])  # Shape: (N,)
        
        # Convert to (N, 3, 32, 32) format (channels first)
        images = images.transpose(0, 3, 1, 2)
        
        # Validate shapes
        expected_size = 50000 if train else 10000
        if len(images) != expected_size:
            raise DataShapeError(f"Expected {expected_size} images, got {len(images)}")
        
        if images.shape[1:] != (3, 32, 32):
            raise DataShapeError(f"Expected image shape (3, 32, 32), got {images.shape[1:]}")
        
        if len(labels) != expected_size:
            raise DataShapeError(f"Expected {expected_size} labels, got {len(labels)}")
        
        logger.info(f"Successfully loaded {len(images)} images with shape {images.shape[1:]}")
        return images, labels
        
    except Exception as e:
        if not download and "not found" in str(e).lower():
            raise FileNotFoundError(f"CIFAR-10 dataset not found in {data_dir}. Set download=True to download it.")
        raise


def load_cifar10h_data(data_dir: str = './cifar-10h-1.0.0/data') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10H dataset (annotator counts and probabilities).
    
    Args:
        data_dir: Directory containing CIFAR-10H .npy files
    
    Returns:
        counts: numpy array of shape (10000, 10) with annotator counts
        probs: numpy array of shape (10000, 10) with probability distributions
    
    Raises:
        FileNotFoundError: If CIFAR-10H files not found
        DataShapeError: If loaded data has unexpected shape
        ValidationError: If data validation fails
    """
    logger.info(f"Loading CIFAR-10H dataset from {data_dir}...")
    
    import os
    
    counts_path = os.path.join(data_dir, 'cifar10h-counts.npy')
    probs_path = os.path.join(data_dir, 'cifar10h-probs.npy')
    
    # Check if files exist
    if not os.path.exists(counts_path):
        logger.error(f"CIFAR-10H counts file not found: {counts_path}")
        raise FileNotFoundError(
            f"CIFAR-10H counts file not found: {counts_path}. "
            f"Please download from https://github.com/jcpeterson/cifar-10h"
        )
    if not os.path.exists(probs_path):
        logger.error(f"CIFAR-10H probabilities file not found: {probs_path}")
        raise FileNotFoundError(
            f"CIFAR-10H probabilities file not found: {probs_path}. "
            f"Please download from https://github.com/jcpeterson/cifar-10h"
        )
    
    logger.debug(f"Loading counts from {counts_path}")
    logger.debug(f"Loading probabilities from {probs_path}")
    
    # Load data
    counts = np.load(counts_path)
    probs = np.load(probs_path)
    
    # Validate shapes
    if counts.shape != (10000, 10):
        logger.error(f"Invalid counts shape: expected (10000, 10), got {counts.shape}")
        raise DataShapeError(f"Expected counts shape (10000, 10), got {counts.shape}")
    if probs.shape != (10000, 10):
        logger.error(f"Invalid probs shape: expected (10000, 10), got {probs.shape}")
        raise DataShapeError(f"Expected probs shape (10000, 10), got {probs.shape}")
    
    # Validate that CIFAR-10H has exactly 10,000 images
    if len(counts) != 10000:
        logger.error(f"CIFAR-10H size mismatch: expected 10000 images, got {len(counts)}")
        raise ValidationError(f"CIFAR-10H must contain exactly 10,000 images, got {len(counts)}")
    
    logger.info(f"Successfully loaded CIFAR-10H data: counts {counts.shape}, probs {probs.shape}")
    logger.debug(f"Counts range: [{counts.min()}, {counts.max()}]")
    logger.debug(f"Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    return counts, probs


def compute_soft_labels(counts: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Compute soft label distributions from annotator counts.
    
    Normalizes annotator counts to create probability distributions and validates
    that all distributions sum to 1.0 within tolerance.
    
    Args:
        counts: numpy array of shape (N, 10) with annotator counts
        epsilon: tolerance for probability sum validation
    
    Returns:
        soft_labels: numpy array of shape (N, 10) with probability distributions
    
    Raises:
        ValidationError: If any distribution doesn't sum to 1.0 within epsilon
        DataShapeError: If counts don't have exactly 10 values per image
    """
    logger.info("Computing soft labels from annotator counts...")
    
    # Validate shape
    if counts.ndim != 2:
        logger.error(f"Invalid counts dimensionality: expected 2D, got {counts.ndim}D")
        raise DataShapeError(f"Expected 2D array, got {counts.ndim}D")
    if counts.shape[1] != 10:
        logger.error(f"Invalid number of classes: expected 10, got {counts.shape[1]}")
        raise DataShapeError(f"Expected 10 classes, got {counts.shape[1]}")
    
    logger.debug(f"Computing probability distributions for {len(counts)} samples")
    
    # Compute probability distributions by normalizing counts
    row_sums = counts.sum(axis=1, keepdims=True)
    
    # Check for zero row sums (potential issue)
    zero_sums = (row_sums == 0).sum()
    if zero_sums > 0:
        logger.warning(f"Found {zero_sums} samples with zero annotator counts")
    
    soft_labels = counts / row_sums
    
    # Validate all distributions sum to 1.0 within epsilon
    sums = soft_labels.sum(axis=1)
    invalid_indices = np.where(np.abs(sums - 1.0) > epsilon)[0]
    
    if len(invalid_indices) > 0:
        first_invalid = invalid_indices[0]
        logger.error(
            f"Soft label validation failed: {len(invalid_indices)} distributions don't sum to 1.0"
        )
        logger.error(f"First invalid index: {first_invalid}, sum: {sums[first_invalid]:.10f}")
        raise ValidationError(
            f"Soft label distribution at index {first_invalid} does not sum to 1.0 "
            f"(sum={sums[first_invalid]:.10f}, epsilon={epsilon})"
        )
    
    logger.info(f"Successfully computed {len(soft_labels)} soft label distributions")
    logger.debug(f"Soft label statistics: min={soft_labels.min():.4f}, max={soft_labels.max():.4f}, mean={soft_labels.mean():.4f}")
    
    return soft_labels


def align_datasets(cifar10_images: np.ndarray, cifar10_labels: np.ndarray,
                   cifar10h_soft_labels: np.ndarray, cifar10h_hard_labels: Optional[np.ndarray] = None) -> List[Tuple]:
    """
    Align CIFAR-10H images with CIFAR-10 test set by index.
    
    Creates a list of (image, soft_label, hard_label) tuples where each CIFAR-10H
    soft label at index i is paired with the CIFAR-10 test image at index i.
    
    Args:
        cifar10_images: numpy array of shape (10000, 3, 32, 32) - CIFAR-10 test images
        cifar10_labels: numpy array of shape (10000,) - CIFAR-10 test hard labels
        cifar10h_soft_labels: numpy array of shape (10000, 10) - CIFAR-10H soft labels
        cifar10h_hard_labels: optional numpy array of shape (10000,) - if provided, use these instead of cifar10_labels
    
    Returns:
        aligned_data: List of (image, soft_label, hard_label) tuples
    
    Raises:
        ValidationError: If dataset sizes don't match or alignment fails
    """
    logger.info("Aligning CIFAR-10H with CIFAR-10 test set...")
    
    # Validate sizes
    if len(cifar10_images) != 10000:
        raise ValidationError(f"CIFAR-10 test set must have 10,000 images, got {len(cifar10_images)}")
    if len(cifar10h_soft_labels) != 10000:
        raise ValidationError(f"CIFAR-10H must have 10,000 soft labels, got {len(cifar10h_soft_labels)}")
    if len(cifar10_labels) != 10000:
        raise ValidationError(f"CIFAR-10 test labels must have 10,000 entries, got {len(cifar10_labels)}")
    
    # Use provided hard labels or CIFAR-10 labels
    hard_labels = cifar10h_hard_labels if cifar10h_hard_labels is not None else cifar10_labels
    
    # Create aligned dataset by pairing by index
    aligned_data = []
    for i in range(10000):
        image = cifar10_images[i]
        soft_label = cifar10h_soft_labels[i]
        hard_label = hard_labels[i]
        
        aligned_data.append((image, soft_label, hard_label))
    
    # Verify alignment preserves index correspondence
    if len(aligned_data) != 10000:
        raise ValidationError(f"Alignment failed: expected 10,000 entries, got {len(aligned_data)}")
    
    logger.info(f"Successfully aligned {len(aligned_data)} images with soft labels")
    return aligned_data


def split_dataset(aligned_data: List[Tuple], random_seed: int = 42) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Split CIFAR-10H into train/val/test with fixed random seed.
    
    Splits the aligned dataset into:
    - Training: 6,000 images (60%)
    - Validation: 2,000 images (20%)
    - Test: 2,000 images (20%)
    
    Args:
        aligned_data: List of (image, soft_label, hard_label) tuples
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_data: List of 6,000 training samples
        val_data: List of 2,000 validation samples
        test_data: List of 2,000 test samples
    
    Raises:
        ValidationError: If splits have overlap or incorrect sizes
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Splitting dataset with random seed {random_seed}...")
    
    if len(aligned_data) != 10000:
        raise ValidationError(f"Expected 10,000 aligned samples, got {len(aligned_data)}")
    
    # Create indices for splitting
    indices = list(range(len(aligned_data)))
    
    # First split: 8000 train+val, 2000 test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=random_seed
    )
    
    # Second split: 6000 train, 2000 val
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.25, random_state=random_seed  # 0.25 * 8000 = 2000
    )
    
    # Create split datasets
    train_data = [aligned_data[i] for i in train_indices]
    val_data = [aligned_data[i] for i in val_indices]
    test_data = [aligned_data[i] for i in test_indices]
    
    # Verify split sizes
    if len(train_data) != 6000:
        raise ValidationError(f"Expected 6,000 training samples, got {len(train_data)}")
    if len(val_data) != 2000:
        raise ValidationError(f"Expected 2,000 validation samples, got {len(val_data)}")
    if len(test_data) != 2000:
        raise ValidationError(f"Expected 2,000 test samples, got {len(test_data)}")
    
    # Verify no overlap between splits
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if len(train_set & val_set) > 0:
        raise ValidationError("Train and validation splits have overlap")
    if len(train_set & test_set) > 0:
        raise ValidationError("Train and test splits have overlap")
    if len(val_set & test_set) > 0:
        raise ValidationError("Validation and test splits have overlap")
    
    # Verify total size
    if len(train_indices) + len(val_indices) + len(test_indices) != 10000:
        raise ValidationError("Split sizes don't sum to 10,000")
    
    logger.info(f"Successfully split dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data


def compute_entropy(probs: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Compute Shannon entropy for probability distributions.
    
    H(p) = -Σ p(y) * log₂(p(y))
    
    Args:
        probs: Array of shape (N, 10) with probability distributions
        epsilon: Small constant for numerical stability
    
    Returns:
        entropies: Array of shape (N,) with entropy values in bits
    
    Raises:
        ValidationError: If entropy values are outside valid range [0, 3.32]
    """
    logger.debug(f"Computing Shannon entropy for {len(probs)} distributions...")
    
    # Add epsilon to avoid log(0)
    probs_safe = probs + epsilon
    # Normalize to ensure sum to 1 after adding epsilon
    probs_safe = probs_safe / probs_safe.sum(axis=1, keepdims=True)
    
    # Compute entropy in bits (log base 2)
    entropies = -np.sum(probs_safe * np.log2(probs_safe), axis=1)
    
    # Verify entropy values are in valid range [0, 3.32] bits
    # Maximum entropy for 10 classes: log₂(10) ≈ 3.32 bits
    max_entropy = np.log2(10)
    if np.any(entropies < 0) or np.any(entropies > max_entropy + 0.01):  # Small tolerance for numerical errors
        invalid_indices = np.where((entropies < 0) | (entropies > max_entropy + 0.01))[0]
        logger.error(
            f"Entropy validation failed: {len(invalid_indices)} values out of range [0, {max_entropy:.2f}]"
        )
        logger.error(f"Invalid indices: {invalid_indices[:5]}... (values: {entropies[invalid_indices[:5]]})")
        raise ValidationError(
            f"Entropy values out of range [0, {max_entropy:.2f}] at indices: {invalid_indices[:5]}... "
            f"(values: {entropies[invalid_indices[:5]]})"
        )
    
    # Log warnings for unusual entropy distributions
    low_entropy_count = (entropies < 0.1).sum()
    high_entropy_count = (entropies > 2.5).sum()
    
    if low_entropy_count > len(entropies) * 0.8:
        logger.warning(f"High proportion of low-entropy samples: {low_entropy_count}/{len(entropies)} ({100*low_entropy_count/len(entropies):.1f}%)")
    
    if high_entropy_count > len(entropies) * 0.2:
        logger.warning(f"High proportion of high-entropy samples: {high_entropy_count}/{len(entropies)} ({100*high_entropy_count/len(entropies):.1f}%)")
    
    logger.debug(f"Computed entropy statistics: min={entropies.min():.3f}, max={entropies.max():.3f}, mean={entropies.mean():.3f}, std={entropies.std():.3f}")
    
    return entropies



# Configuration serialization
@dataclass
class DataPipelineConfig:
    """
    Configuration for data pipeline parameters.
    
    Attributes:
        cifar10_data_dir: Directory for CIFAR-10 dataset
        cifar10h_data_dir: Directory for CIFAR-10H dataset
        train_size: Number of training samples (default: 6000)
        val_size: Number of validation samples (default: 2000)
        test_size: Number of test samples (default: 2000)
        random_seed: Random seed for splitting (default: 42)
        epsilon: Numerical stability constant (default: 1e-7)
    """
    cifar10_data_dir: str = './data'
    cifar10h_data_dir: str = './cifar-10h-1.0.0/data'
    train_size: int = 6000
    val_size: int = 2000
    test_size: int = 2000
    random_seed: int = 42
    epsilon: float = 1e-7
    
    def to_json(self, filepath: str):
        """
        Serialize configuration to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Saved data pipeline configuration to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> 'DataPipelineConfig':
        """
        Deserialize configuration from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            DataPipelineConfig instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If JSON is invalid
        """
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            config = cls(**data)
            config.validate()
            
            logger.info(f"Loaded data pipeline configuration from {filepath}")
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigParseError(f"Invalid JSON in configuration file: {e}")
        except TypeError as e:
            raise ConfigParseError(f"Invalid configuration fields: {e}")
    
    def validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValidationError: If any parameter is invalid
        """
        if self.train_size + self.val_size + self.test_size != 10000:
            raise ValidationError(
                f"Split sizes must sum to 10000, got {self.train_size + self.val_size + self.test_size}"
            )
        
        if self.train_size <= 0 or self.val_size <= 0 or self.test_size <= 0:
            raise ValidationError("All split sizes must be positive")
        
        if self.epsilon <= 0 or self.epsilon >= 1e-5:
            raise ValidationError(f"Epsilon must be in range (0, 1e-5), got {self.epsilon}")
        
        if self.random_seed < 0:
            raise ValidationError(f"Random seed must be non-negative, got {self.random_seed}")
    
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
                "cifar10_data_dir": {"type": "string"},
                "cifar10h_data_dir": {"type": "string"},
                "train_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "val_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "test_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "random_seed": {"type": "integer", "minimum": 0},
                "epsilon": {"type": "number", "minimum": 0, "exclusiveMaximum": 1e-5}
            },
            "required": ["cifar10_data_dir", "cifar10h_data_dir", "train_size", 
                        "val_size", "test_size", "random_seed", "epsilon"]
        }
