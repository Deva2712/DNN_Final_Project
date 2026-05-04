"""
Shared pytest fixtures and configuration for CIFAR-10 Human Disagreement Predictor tests.

This module provides:
- Hypothesis profile configuration (CI vs dev)
- Mock dataset fixtures
- Model fixtures
- Temporary directory fixtures
- Sample probability distribution fixtures
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, List

import pytest
import numpy as np
import torch
import torch.nn as nn
from hypothesis import settings

# Configure Hypothesis profiles
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)

# Load profile from environment variable or default to CI
profile = os.getenv("HYPOTHESIS_PROFILE", "ci")
settings.load_profile(profile)


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def mock_cifar10h_dataset():
    """
    Create a small mock CIFAR-10H dataset for testing.
    
    Returns:
        dict: Contains images, soft_labels, hard_labels, entropies
            - images: (100, 3, 32, 32) tensor
            - soft_labels: (100, 10) tensor with probability distributions
            - hard_labels: (100,) tensor with class labels
            - entropies: (100,) tensor with Shannon entropy values
    """
    num_samples = 100
    num_classes = 10
    
    # Generate random images
    images = torch.randn(num_samples, 3, 32, 32)
    
    # Generate random soft labels (probability distributions)
    # Use Dirichlet distribution to ensure valid probabilities
    alpha = np.ones(num_classes)
    soft_labels = np.random.dirichlet(alpha, size=num_samples)
    soft_labels = torch.from_numpy(soft_labels).float()
    
    # Generate hard labels (argmax of soft labels)
    hard_labels = torch.argmax(soft_labels, dim=1)
    
    # Compute entropies
    epsilon = 1e-7
    probs_safe = soft_labels + epsilon
    probs_safe = probs_safe / probs_safe.sum(dim=1, keepdim=True)
    entropies = -(probs_safe * torch.log2(probs_safe)).sum(dim=1)
    
    return {
        'images': images,
        'soft_labels': soft_labels,
        'hard_labels': hard_labels,
        'entropies': entropies
    }


@pytest.fixture
def mock_cifar10h_splits(mock_cifar10h_dataset):
    """
    Create train/val/test splits from mock dataset.
    
    Returns:
        dict: Contains train, val, test splits with indices
    """
    num_samples = len(mock_cifar10h_dataset['images'])
    indices = np.arange(num_samples)
    
    # Split: 60% train, 20% val, 20% test
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'dataset': mock_cifar10h_dataset
    }


@pytest.fixture
def sample_probability_distributions():
    """
    Generate sample probability distributions for testing.
    
    Returns:
        dict: Contains various types of probability distributions
            - uniform: Uniform distribution over 10 classes
            - peaked: Highly peaked distribution (low entropy)
            - flat: Flat distribution (high entropy)
            - sparse: Distribution with zeros
            - batch: Batch of random distributions
    """
    num_classes = 10
    
    # Uniform distribution
    uniform = torch.ones(num_classes) / num_classes
    
    # Peaked distribution (low entropy)
    peaked = torch.zeros(num_classes)
    peaked[0] = 0.9
    peaked[1:] = 0.1 / (num_classes - 1)
    
    # Flat distribution (high entropy)
    flat = torch.ones(num_classes) / num_classes
    
    # Sparse distribution (with zeros)
    sparse = torch.zeros(num_classes)
    sparse[0] = 0.5
    sparse[1] = 0.3
    sparse[2] = 0.2
    
    # Batch of random distributions
    batch_size = 32
    alpha = np.ones(num_classes)
    batch = np.random.dirichlet(alpha, size=batch_size)
    batch = torch.from_numpy(batch).float()
    
    return {
        'uniform': uniform,
        'peaked': peaked,
        'flat': flat,
        'sparse': sparse,
        'batch': batch
    }


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def small_test_model():
    """
    Create a small model for testing (faster than full ResNet-18).
    
    Returns:
        nn.Module: Small CNN model with same interface as DisagreementPredictor
    """
    class SmallTestModel(nn.Module):
        """Lightweight model for testing."""
        
        def __init__(self, num_classes=10):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
        
        def get_features(self, x):
            return self.backbone(x)
    
    return SmallTestModel()


@pytest.fixture
def pretrained_test_model(small_test_model):
    """
    Create a small model with pretrained weights (random initialization).
    
    Returns:
        nn.Module: Small model with initialized weights
    """
    # Initialize weights
    for module in small_test_model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return small_test_model


# ============================================================================
# Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """
    Create a temporary directory for test outputs.
    
    Yields:
        Path: Path to temporary directory
    
    Cleanup:
        Removes directory after test completes
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_checkpoint_dir():
    """
    Create a temporary directory for model checkpoints.
    
    Yields:
        Path: Path to temporary checkpoint directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        yield checkpoint_dir


@pytest.fixture
def temp_data_dir():
    """
    Create a temporary directory for test data files.
    
    Yields:
        Path: Path to temporary data directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        yield data_dir


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_data_pipeline_config():
    """
    Create a sample DataPipelineConfig for testing.
    
    Returns:
        dict: Configuration parameters
    """
    return {
        'cifar10_root': './data/cifar10',
        'cifar10h_counts_path': 'cifar-10h-1.0.0/data/cifar10h-counts.npy',
        'cifar10h_probs_path': 'cifar-10h-1.0.0/data/cifar10h-probs.npy',
        'train_size': 6000,
        'val_size': 2000,
        'test_size': 2000,
        'random_seed': 42,
        'epsilon': 1e-7
    }


@pytest.fixture
def sample_model_config():
    """
    Create a sample ModelConfig for testing.
    
    Returns:
        dict: Configuration parameters
    """
    return {
        'backbone': 'resnet18',
        'backbone_pretrained': False,
        'head_hidden_dim': 256,
        'num_classes': 10
    }


@pytest.fixture
def sample_training_config():
    """
    Create a sample TrainingConfig for testing.
    
    Returns:
        dict: Configuration parameters
    """
    return {
        'pretrain_lr': 1e-3,
        'pretrain_epochs': 100,
        'pretrain_batch_size': 128,
        'finetune_lr': 1e-4,
        'finetune_epochs': 50,
        'finetune_batch_size': 64,
        'loss_function': 'kl',
        'lambda_entropy': 0.1,
        'optimizer': 'adamw',
        'weight_decay': 1e-4,
        'patience': 10,
        'use_augmentation': True,
        'random_flip_prob': 0.5,
        'random_crop_padding': 4,
        'random_seed': 42
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def cifar10_class_names():
    """
    Get CIFAR-10 class names.
    
    Returns:
        list: List of 10 class names
    """
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


@pytest.fixture
def set_random_seed():
    """
    Set random seed for reproducibility in tests.
    
    Returns:
        callable: Function to set seed
    """
    def _set_seed(seed=42):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    return _set_seed


@pytest.fixture
def device():
    """
    Get device for testing (CPU or CUDA if available).
    
    Returns:
        str: 'cuda' if available, else 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Parametrize Helpers
# ============================================================================

@pytest.fixture
def entropy_test_cases():
    """
    Generate test cases for entropy computation.
    
    Returns:
        list: List of (distribution, expected_entropy) tuples
    """
    return [
        # Uniform distribution: H = log2(10) ≈ 3.32
        (np.ones(10) / 10, 3.32),
        # Peaked distribution: H ≈ 0
        (np.array([0.99] + [0.01/9]*9), 0.0),
        # Two-class uniform: H = 1.0
        (np.array([0.5, 0.5] + [0.0]*8), 1.0),
    ]


@pytest.fixture
def kl_divergence_test_cases():
    """
    Generate test cases for KL divergence computation.
    
    Returns:
        list: List of (p, q, expected_kl) tuples
    """
    return [
        # Identical distributions: KL = 0
        (np.ones(10) / 10, np.ones(10) / 10, 0.0),
        # Different distributions: KL > 0
        (np.array([0.9, 0.1]), np.array([0.1, 0.9]), None),  # Compute expected
    ]


# ============================================================================
# Session-scoped Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_cache_dir(tmp_path_factory):
    """
    Create a session-scoped temporary directory for caching test data.
    
    This is useful for expensive data generation that can be reused across tests.
    
    Returns:
        Path: Path to cache directory
    """
    cache_dir = tmp_path_factory.mktemp("test_data_cache")
    return cache_dir


# ============================================================================
# Hooks for Test Reporting
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # Add custom markers
    config.addinivalue_line(
        "markers", "property: Property-based tests using Hypothesis"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for complete workflows"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    """
    for item in items:
        # Add property marker to tests using Hypothesis
        if "hypothesis" in item.keywords:
            item.add_marker(pytest.mark.property)
        
        # Add markers based on test file location
        if "property_tests" in str(item.fspath):
            item.add_marker(pytest.mark.property)
        elif "unit_tests" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
