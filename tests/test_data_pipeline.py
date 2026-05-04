"""
Unit Tests for Data Pipeline Module

Tests data loading, soft label computation, dataset alignment, splitting,
entropy computation, and configuration serialization.
"""

import pytest
import numpy as np
import torch
import os
import tempfile
from pathlib import Path

from src.data_pipeline import (
    load_cifar10_data,
    load_cifar10h_data,
    compute_soft_labels,
    align_datasets,
    split_dataset,
    compute_entropy,
    CIFAR10HDataset,
    DataPipelineConfig,
    ValidationError,
    DataShapeError
)


class TestDataLoading:
    """Tests for CIFAR-10 and CIFAR-10H data loading."""
    
    def test_load_cifar10_test_data(self):
        """Test loading CIFAR-10 test set."""
        images, labels = load_cifar10_data(train=False, download=True)
        
        # Verify size
        assert len(images) == 10000, "CIFAR-10 test set should have 10,000 images"
        assert len(labels) == 10000, "CIFAR-10 test set should have 10,000 labels"
        
        # Verify shapes
        assert images.shape == (10000, 3, 32, 32), f"Expected shape (10000, 3, 32, 32), got {images.shape}"
        assert labels.shape == (10000,), f"Expected shape (10000,), got {labels.shape}"
        
        # Verify data types
        assert images.dtype == np.uint8, f"Expected uint8, got {images.dtype}"
        assert labels.dtype in [np.int64, np.int32], f"Expected int type, got {labels.dtype}"
        
        # Verify label range
        assert np.all(labels >= 0) and np.all(labels <= 9), "Labels should be in range [0, 9]"
    
    def test_load_cifar10_train_data(self):
        """Test loading CIFAR-10 training set."""
        images, labels = load_cifar10_data(train=True, download=True)
        
        # Verify size
        assert len(images) == 50000, "CIFAR-10 training set should have 50,000 images"
        assert len(labels) == 50000, "CIFAR-10 training set should have 50,000 labels"
        
        # Verify shapes
        assert images.shape == (50000, 3, 32, 32), f"Expected shape (50000, 3, 32, 32), got {images.shape}"
    
    def test_load_cifar10h_data(self):
        """Test loading CIFAR-10H data."""
        counts, probs = load_cifar10h_data()
        
        # Verify sizes
        assert len(counts) == 10000, "CIFAR-10H should have 10,000 images"
        assert len(probs) == 10000, "CIFAR-10H should have 10,000 probability distributions"
        
        # Verify shapes
        assert counts.shape == (10000, 10), f"Expected counts shape (10000, 10), got {counts.shape}"
        assert probs.shape == (10000, 10), f"Expected probs shape (10000, 10), got {probs.shape}"
        
        # Verify probabilities sum to 1
        prob_sums = probs.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6), "Probabilities should sum to 1.0"
    
    def test_load_cifar10h_missing_files(self):
        """Test error handling for missing CIFAR-10H files."""
        with pytest.raises(FileNotFoundError):
            load_cifar10h_data(data_dir='/nonexistent/path')


class TestSoftLabelComputation:
    """Tests for soft label computation and validation."""
    
    def test_compute_soft_labels_basic(self):
        """Test basic soft label computation."""
        # Create sample counts
        counts = np.array([
            [10, 5, 3, 2, 0, 0, 0, 0, 0, 0],  # 20 total
            [0, 0, 0, 0, 15, 10, 5, 0, 0, 0],  # 30 total
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]     # 50 total
        ], dtype=float)
        
        soft_labels = compute_soft_labels(counts)
        
        # Verify shape
        assert soft_labels.shape == (3, 10), f"Expected shape (3, 10), got {soft_labels.shape}"
        
        # Verify normalization
        sums = soft_labels.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-7), "Soft labels should sum to 1.0"
        
        # Verify specific values
        expected_first = np.array([10, 5, 3, 2, 0, 0, 0, 0, 0, 0]) / 20
        assert np.allclose(soft_labels[0], expected_first, atol=1e-7)
    
    def test_compute_soft_labels_validation(self):
        """Test that soft labels are properly validated."""
        counts = np.array([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], dtype=float)
        soft_labels = compute_soft_labels(counts)
        
        # Should be uniform distribution
        expected = np.ones(10) / 10
        assert np.allclose(soft_labels[0], expected, atol=1e-7)
    
    def test_compute_soft_labels_wrong_shape(self):
        """Test error handling for wrong input shape."""
        # 1D array
        with pytest.raises(DataShapeError):
            compute_soft_labels(np.array([1, 2, 3, 4, 5]))
        
        # Wrong number of classes
        with pytest.raises(DataShapeError):
            compute_soft_labels(np.array([[1, 2, 3, 4, 5]]))


class TestDatasetAlignment:
    """Tests for dataset alignment."""
    
    def test_align_datasets_basic(self):
        """Test basic dataset alignment."""
        # Create mock data
        images = np.random.rand(10000, 3, 32, 32).astype(np.float32)
        hard_labels = np.random.randint(0, 10, size=10000)
        soft_labels = np.random.dirichlet(np.ones(10), size=10000)
        
        aligned = align_datasets(images, hard_labels, soft_labels)
        
        # Verify size
        assert len(aligned) == 10000, f"Expected 10,000 aligned samples, got {len(aligned)}"
        
        # Verify structure
        for i in range(10):
            image, soft_label, hard_label = aligned[i]
            assert image.shape == (3, 32, 32)
            assert soft_label.shape == (10,)
            assert isinstance(hard_label, (int, np.integer))
    
    def test_align_datasets_index_correspondence(self):
        """Test that alignment preserves index correspondence."""
        # Create identifiable data
        images = np.arange(10000 * 3 * 32 * 32).reshape(10000, 3, 32, 32).astype(np.float32)
        hard_labels = np.arange(10000) % 10
        soft_labels = np.eye(10)[hard_labels]  # One-hot encoding
        
        aligned = align_datasets(images, hard_labels, soft_labels)
        
        # Verify alignment
        for i in range(100):  # Check first 100
            image, soft_label, hard_label = aligned[i]
            assert np.array_equal(image, images[i])
            assert np.array_equal(soft_label, soft_labels[i])
            assert hard_label == hard_labels[i]
    
    def test_align_datasets_size_mismatch(self):
        """Test error handling for size mismatches."""
        images = np.random.rand(9999, 3, 32, 32).astype(np.float32)
        hard_labels = np.random.randint(0, 10, size=10000)
        soft_labels = np.random.dirichlet(np.ones(10), size=10000)
        
        with pytest.raises(ValidationError):
            align_datasets(images, hard_labels, soft_labels)


class TestDatasetSplitting:
    """Tests for dataset splitting."""
    
    def test_split_dataset_sizes(self):
        """Test that split produces correct sizes."""
        # Create mock aligned data
        aligned_data = [(np.random.rand(3, 32, 32), np.random.rand(10), i % 10) 
                       for i in range(10000)]
        
        train, val, test = split_dataset(aligned_data, random_seed=42)
        
        # Verify sizes
        assert len(train) == 6000, f"Expected 6,000 training samples, got {len(train)}"
        assert len(val) == 2000, f"Expected 2,000 validation samples, got {len(val)}"
        assert len(test) == 2000, f"Expected 2,000 test samples, got {len(test)}"
    
    def test_split_dataset_no_overlap(self):
        """Test that splits have no overlap."""
        # Create identifiable data
        aligned_data = [(np.array([[[i]]]), np.random.rand(10), i % 10) 
                       for i in range(10000)]
        
        train, val, test = split_dataset(aligned_data, random_seed=42)
        
        # Extract identifiers
        train_ids = set([int(img[0, 0, 0]) for img, _, _ in train])
        val_ids = set([int(img[0, 0, 0]) for img, _, _ in val])
        test_ids = set([int(img[0, 0, 0]) for img, _, _ in test])
        
        # Verify no overlap
        assert len(train_ids & val_ids) == 0, "Train and val splits should not overlap"
        assert len(train_ids & test_ids) == 0, "Train and test splits should not overlap"
        assert len(val_ids & test_ids) == 0, "Val and test splits should not overlap"
        
        # Verify total coverage
        assert len(train_ids | val_ids | test_ids) == 10000, "Splits should cover all data"
    
    def test_split_dataset_reproducibility(self):
        """Test that splitting is reproducible with same seed."""
        aligned_data = [(np.random.rand(3, 32, 32), np.random.rand(10), i % 10) 
                       for i in range(10000)]
        
        train1, val1, test1 = split_dataset(aligned_data, random_seed=42)
        train2, val2, test2 = split_dataset(aligned_data, random_seed=42)
        
        # Compare first elements (should be identical)
        assert np.array_equal(train1[0][0], train2[0][0])
        assert np.array_equal(val1[0][0], val2[0][0])
        assert np.array_equal(test1[0][0], test2[0][0])


class TestEntropyComputation:
    """Tests for Shannon entropy computation."""
    
    def test_compute_entropy_uniform(self):
        """Test entropy for uniform distribution."""
        # Uniform distribution has maximum entropy
        probs = np.ones((1, 10)) / 10
        entropy = compute_entropy(probs)
        
        expected = np.log2(10)  # Maximum entropy for 10 classes
        assert np.isclose(entropy[0], expected, atol=1e-6), \
            f"Expected entropy {expected:.4f}, got {entropy[0]:.4f}"
    
    def test_compute_entropy_deterministic(self):
        """Test entropy for deterministic distribution."""
        # One-hot distribution has zero entropy
        probs = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        entropy = compute_entropy(probs)
        
        assert entropy[0] < 0.01, f"Expected near-zero entropy, got {entropy[0]:.4f}"
    
    def test_compute_entropy_range(self):
        """Test that entropy values are in valid range."""
        # Random distributions
        probs = np.random.dirichlet(np.ones(10), size=100)
        entropies = compute_entropy(probs)
        
        max_entropy = np.log2(10)
        assert np.all(entropies >= 0), "Entropy should be non-negative"
        assert np.all(entropies <= max_entropy + 0.01), \
            f"Entropy should not exceed {max_entropy:.2f} bits"
    
    def test_compute_entropy_numerical_stability(self):
        """Test numerical stability with near-zero probabilities."""
        # Distribution with very small probabilities
        probs = np.array([[0.9, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        entropy = compute_entropy(probs, epsilon=1e-7)
        
        # Should not produce NaN or Inf
        assert np.isfinite(entropy[0]), "Entropy should be finite"
        assert entropy[0] >= 0, "Entropy should be non-negative"


class TestCIFAR10HDataset:
    """Tests for custom PyTorch Dataset class."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        images = torch.randn(100, 3, 32, 32)
        soft_labels = torch.rand(100, 10)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        hard_labels = torch.randint(0, 10, (100,))
        entropies = torch.rand(100)
        
        dataset = CIFAR10HDataset(images, soft_labels, hard_labels, entropies)
        
        assert len(dataset) == 100, f"Expected length 100, got {len(dataset)}"
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        images = torch.randn(100, 3, 32, 32)
        soft_labels = torch.rand(100, 10)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        hard_labels = torch.randint(0, 10, (100,))
        entropies = torch.rand(100)
        
        dataset = CIFAR10HDataset(images, soft_labels, hard_labels, entropies)
        
        # Get single item
        image, soft_label, hard_label, entropy = dataset[0]
        
        assert image.shape == (3, 32, 32), f"Expected image shape (3, 32, 32), got {image.shape}"
        assert soft_label.shape == (10,), f"Expected soft_label shape (10,), got {soft_label.shape}"
        assert isinstance(hard_label, torch.Tensor), "hard_label should be a tensor"
        assert isinstance(entropy, torch.Tensor), "entropy should be a tensor"
    
    def test_dataset_with_transform(self):
        """Test dataset with transforms."""
        from torchvision import transforms
        
        images = torch.randn(100, 3, 32, 32)
        soft_labels = torch.rand(100, 10)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        hard_labels = torch.randint(0, 10, (100,))
        entropies = torch.rand(100)
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)  # Always flip for testing
        ])
        
        dataset = CIFAR10HDataset(images, soft_labels, hard_labels, entropies, transform=transform)
        
        # Get item (should be transformed)
        image, _, _, _ = dataset[0]
        assert image.shape == (3, 32, 32), "Transform should preserve shape"


class TestDataPipelineConfig:
    """Tests for configuration serialization."""
    
    def test_config_default_values(self):
        """Test default configuration values."""
        config = DataPipelineConfig()
        
        assert config.train_size == 6000
        assert config.val_size == 2000
        assert config.test_size == 2000
        assert config.random_seed == 42
        assert config.epsilon == 1e-7
    
    def test_config_validation_valid(self):
        """Test validation with valid configuration."""
        config = DataPipelineConfig()
        config.validate()  # Should not raise
    
    def test_config_validation_invalid_sizes(self):
        """Test validation with invalid split sizes."""
        config = DataPipelineConfig(train_size=5000, val_size=2000, test_size=2000)
        
        with pytest.raises(ValidationError):
            config.validate()
    
    def test_config_validation_negative_size(self):
        """Test validation with negative size."""
        config = DataPipelineConfig(train_size=-1, val_size=2000, test_size=8001)
        
        with pytest.raises(ValidationError):
            config.validate()
    
    def test_config_to_json(self):
        """Test serialization to JSON."""
        config = DataPipelineConfig(
            cifar10_data_dir='./test_data',
            random_seed=123
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_json(temp_path)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data['cifar10_data_dir'] == './test_data'
            assert data['random_seed'] == 123
            assert data['train_size'] == 6000
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_config_from_json(self):
        """Test deserialization from JSON."""
        config_data = {
            'cifar10_data_dir': './test_data',
            'cifar10h_data_dir': './test_cifar10h',
            'train_size': 6000,
            'val_size': 2000,
            'test_size': 2000,
            'random_seed': 123,
            'epsilon': 1e-7
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = DataPipelineConfig.from_json(temp_path)
            
            assert config.cifar10_data_dir == './test_data'
            assert config.random_seed == 123
            assert config.train_size == 6000
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_config_round_trip(self):
        """Test round-trip serialization (to_json -> from_json)."""
        original = DataPipelineConfig(
            cifar10_data_dir='./custom_data',
            random_seed=999
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            original.to_json(temp_path)
            loaded = DataPipelineConfig.from_json(temp_path)
            
            assert loaded.cifar10_data_dir == original.cifar10_data_dir
            assert loaded.random_seed == original.random_seed
            assert loaded.train_size == original.train_size
            assert loaded.epsilon == original.epsilon
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_config_from_json_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            DataPipelineConfig.from_json('/nonexistent/config.json')
    
    def test_config_json_schema(self):
        """Test JSON schema generation."""
        schema = DataPipelineConfig.get_json_schema()
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'train_size' in schema['properties']
        assert 'random_seed' in schema['properties']
from hypothesis import given, strategies as st
from hypothesis import assume

import json



class TestProbabilityDistributionNormalization:
    """
    Property-based tests for probability distribution normalization.
    
    **Validates: Requirements 2.1, 2.2**
    """
    
    @given(
        counts=st.lists(
            st.lists(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=10,
                max_size=10
            ),
            min_size=1,
            max_size=100
        )
    )
    def test_property_normalized_distributions_sum_to_one(self, counts):
        """
        **Property 1: Probability Distribution Normalization**
        
        FOR ALL count arrays with non-negative values,
        WHEN normalized to probability distributions,
        THEN the sum of probabilities MUST equal 1.0 within epsilon=1e-7.
        
        **Validates: Requirements 2.1, 2.2**
        """
        # Convert to numpy array
        counts_array = np.array(counts, dtype=float)
        
        # Filter out rows where all counts are zero (would cause division by zero)
        row_sums = counts_array.sum(axis=1)
        assume(np.all(row_sums > 0))
        
        # Compute soft labels using the function under test
        soft_labels = compute_soft_labels(counts_array, epsilon=1e-7)
        
        # Property: All distributions must sum to 1.0 within epsilon
        epsilon = 1e-7
        sums = soft_labels.sum(axis=1)
        
        # Verify all sums are within tolerance
        assert np.all(np.abs(sums - 1.0) <= epsilon), \
            f"Some distributions don't sum to 1.0: min={sums.min()}, max={sums.max()}"
        
        # Additional checks
        assert soft_labels.shape == counts_array.shape, \
            "Output shape must match input shape"
        assert np.all(soft_labels >= 0), \
            "All probabilities must be non-negative"
        assert np.all(soft_labels <= 1.0), \
            "All probabilities must be <= 1.0"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=50),
        num_classes=st.just(10)  # CIFAR-10 has 10 classes
    )
    def test_property_uniform_counts_produce_uniform_distribution(self, batch_size, num_classes):
        """
        **Property: Uniform Counts Produce Uniform Distribution**
        
        FOR ALL uniform count arrays (all counts equal),
        WHEN normalized to probability distributions,
        THEN each probability MUST equal 1/num_classes.
        
        **Validates: Requirement 2.1**
        """
        # Create uniform counts (all equal)
        uniform_value = 10.0
        counts = np.full((batch_size, num_classes), uniform_value, dtype=float)
        
        # Compute soft labels
        soft_labels = compute_soft_labels(counts, epsilon=1e-7)
        
        # Property: All probabilities should be equal to 1/num_classes
        expected_prob = 1.0 / num_classes
        
        assert np.allclose(soft_labels, expected_prob, atol=1e-6), \
            f"Uniform counts should produce uniform distribution {expected_prob}"
    
    @given(
        counts=st.lists(
            st.lists(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=10,
                max_size=10
            ),
            min_size=1,
            max_size=50
        )
    )
    def test_property_normalization_preserves_relative_proportions(self, counts):
        """
        **Property: Normalization Preserves Relative Proportions**
        
        FOR ALL count arrays,
        WHEN normalized to probability distributions,
        THEN the relative ordering of probabilities MUST match the relative ordering of counts.
        
        **Validates: Requirement 2.1**
        """
        counts_array = np.array(counts, dtype=float)
        
        # Filter out rows where all counts are zero
        row_sums = counts_array.sum(axis=1)
        assume(np.all(row_sums > 0))
        
        # Compute soft labels
        soft_labels = compute_soft_labels(counts_array, epsilon=1e-7)
        
        # Property: For each row, the ordering should be preserved
        for i in range(len(counts_array)):
            count_order = np.argsort(counts_array[i])
            prob_order = np.argsort(soft_labels[i])
            
            # The orderings should match
            assert np.array_equal(count_order, prob_order), \
                f"Row {i}: Normalization changed relative ordering"
    
    @given(
        counts=st.lists(
            st.lists(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=10,
                max_size=10
            ),
            min_size=1,
            max_size=50
        ),
        scale_factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    def test_property_scaling_invariance(self, counts, scale_factor):
        """
        **Property: Scaling Invariance**
        
        FOR ALL count arrays and positive scale factors,
        WHEN counts are scaled by a constant factor before normalization,
        THEN the resulting probability distribution MUST be identical.
        
        **Validates: Requirement 2.1**
        """
        counts_array = np.array(counts, dtype=float)
        
        # Filter out rows where all counts are zero
        row_sums = counts_array.sum(axis=1)
        assume(np.all(row_sums > 0))
        
        # Compute soft labels for original counts
        soft_labels_original = compute_soft_labels(counts_array, epsilon=1e-7)
        
        # Scale counts and compute soft labels
        scaled_counts = counts_array * scale_factor
        soft_labels_scaled = compute_soft_labels(scaled_counts, epsilon=1e-7)
        
        # Property: Distributions should be identical (scaling doesn't affect normalization)
        assert np.allclose(soft_labels_original, soft_labels_scaled, atol=1e-6), \
            "Scaling counts should not affect normalized distribution"
    
    def test_property_deterministic_distribution_has_one_probability(self):
        """
        **Property: Deterministic Distribution**
        
        WHEN all counts are zero except one class,
        THEN that class MUST have probability 1.0 and all others 0.0.
        
        **Validates: Requirement 2.1**
        """
        # Create deterministic counts (only one class has counts)
        for target_class in range(10):
            counts = np.zeros((1, 10), dtype=float)
            counts[0, target_class] = 50.0
            
            # Compute soft labels
            soft_labels = compute_soft_labels(counts, epsilon=1e-7)
            
            # Property: Target class should have probability ~1.0
            assert soft_labels[0, target_class] > 0.99, \
                f"Class {target_class} should have probability ~1.0"
            
            # Other classes should have probability ~0.0
            for other_class in range(10):
                if other_class != target_class:
                    assert soft_labels[0, other_class] < 0.01, \
                        f"Class {other_class} should have probability ~0.0"
