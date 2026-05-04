"""
Verification tests for shared fixtures.

These tests ensure that all fixtures in conftest.py work correctly.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestDatasetFixtures:
    """Test dataset fixtures."""
    
    def test_mock_cifar10h_dataset(self, mock_cifar10h_dataset):
        """Verify mock dataset fixture."""
        assert 'images' in mock_cifar10h_dataset
        assert 'soft_labels' in mock_cifar10h_dataset
        assert 'hard_labels' in mock_cifar10h_dataset
        assert 'entropies' in mock_cifar10h_dataset
        
        images = mock_cifar10h_dataset['images']
        soft_labels = mock_cifar10h_dataset['soft_labels']
        hard_labels = mock_cifar10h_dataset['hard_labels']
        entropies = mock_cifar10h_dataset['entropies']
        
        # Check shapes
        assert images.shape == (100, 3, 32, 32)
        assert soft_labels.shape == (100, 10)
        assert hard_labels.shape == (100,)
        assert entropies.shape == (100,)
        
        # Check soft labels are valid probability distributions
        assert torch.allclose(soft_labels.sum(dim=1), torch.ones(100), atol=1e-6)
        assert (soft_labels >= 0).all()
        assert (soft_labels <= 1).all()
        
        # Check entropies are in valid range [0, 3.32]
        assert (entropies >= 0).all()
        assert (entropies <= 3.32).all()
    
    def test_mock_cifar10h_splits(self, mock_cifar10h_splits):
        """Verify dataset splits fixture."""
        assert 'train_indices' in mock_cifar10h_splits
        assert 'val_indices' in mock_cifar10h_splits
        assert 'test_indices' in mock_cifar10h_splits
        assert 'dataset' in mock_cifar10h_splits
        
        train_indices = mock_cifar10h_splits['train_indices']
        val_indices = mock_cifar10h_splits['val_indices']
        test_indices = mock_cifar10h_splits['test_indices']
        
        # Check sizes (60/20/20 split of 100 samples)
        assert len(train_indices) == 60
        assert len(val_indices) == 20
        assert len(test_indices) == 20
        
        # Check disjointness
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
    
    def test_sample_probability_distributions(self, sample_probability_distributions):
        """Verify probability distributions fixture."""
        assert 'uniform' in sample_probability_distributions
        assert 'peaked' in sample_probability_distributions
        assert 'flat' in sample_probability_distributions
        assert 'sparse' in sample_probability_distributions
        assert 'batch' in sample_probability_distributions
        
        uniform = sample_probability_distributions['uniform']
        peaked = sample_probability_distributions['peaked']
        batch = sample_probability_distributions['batch']
        
        # Check shapes
        assert uniform.shape == (10,)
        assert peaked.shape == (10,)
        assert batch.shape == (32, 10)
        
        # Check all are valid probability distributions
        assert torch.allclose(uniform.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(peaked.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(batch.sum(dim=1), torch.ones(32), atol=1e-6)


class TestModelFixtures:
    """Test model fixtures."""
    
    def test_small_test_model(self, small_test_model):
        """Verify small test model fixture."""
        # Check model has required methods
        assert hasattr(small_test_model, 'forward')
        assert hasattr(small_test_model, 'get_features')
        
        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        output = small_test_model(x)
        
        # Check output shape and properties
        assert output.shape == (4, 10)
        assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-6)
        assert (output >= 0).all()
        assert (output <= 1).all()
        
        # Test feature extraction
        features = small_test_model.get_features(x)
        assert features.shape == (4, 32)
    
    def test_pretrained_test_model(self, pretrained_test_model):
        """Verify pretrained test model fixture."""
        # Check weights are initialized (not all zeros)
        has_nonzero_weights = False
        for param in pretrained_test_model.parameters():
            if param.abs().sum() > 0:
                has_nonzero_weights = True
                break
        
        assert has_nonzero_weights, "Model weights should be initialized"


class TestDirectoryFixtures:
    """Test directory fixtures."""
    
    def test_temp_output_dir(self, temp_output_dir):
        """Verify temporary output directory fixture."""
        assert isinstance(temp_output_dir, Path)
        assert temp_output_dir.exists()
        assert temp_output_dir.is_dir()
        
        # Test writing to directory
        test_file = temp_output_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
    
    def test_temp_checkpoint_dir(self, temp_checkpoint_dir):
        """Verify temporary checkpoint directory fixture."""
        assert isinstance(temp_checkpoint_dir, Path)
        assert temp_checkpoint_dir.exists()
        assert temp_checkpoint_dir.is_dir()
        assert temp_checkpoint_dir.name == "checkpoints"
    
    def test_temp_data_dir(self, temp_data_dir):
        """Verify temporary data directory fixture."""
        assert isinstance(temp_data_dir, Path)
        assert temp_data_dir.exists()
        assert temp_data_dir.is_dir()
        assert temp_data_dir.name == "data"


class TestConfigurationFixtures:
    """Test configuration fixtures."""
    
    def test_sample_data_pipeline_config(self, sample_data_pipeline_config):
        """Verify data pipeline config fixture."""
        assert 'cifar10_root' in sample_data_pipeline_config
        assert 'cifar10h_counts_path' in sample_data_pipeline_config
        assert 'train_size' in sample_data_pipeline_config
        assert 'val_size' in sample_data_pipeline_config
        assert 'test_size' in sample_data_pipeline_config
        assert 'random_seed' in sample_data_pipeline_config
        assert 'epsilon' in sample_data_pipeline_config
        
        # Check values
        assert sample_data_pipeline_config['train_size'] == 6000
        assert sample_data_pipeline_config['val_size'] == 2000
        assert sample_data_pipeline_config['test_size'] == 2000
        assert sample_data_pipeline_config['random_seed'] == 42
    
    def test_sample_model_config(self, sample_model_config):
        """Verify model config fixture."""
        assert 'backbone' in sample_model_config
        assert 'backbone_pretrained' in sample_model_config
        assert 'head_hidden_dim' in sample_model_config
        assert 'num_classes' in sample_model_config
        
        # Check values
        assert sample_model_config['backbone'] == 'resnet18'
        assert sample_model_config['head_hidden_dim'] == 256
        assert sample_model_config['num_classes'] == 10
    
    def test_sample_training_config(self, sample_training_config):
        """Verify training config fixture."""
        assert 'pretrain_lr' in sample_training_config
        assert 'finetune_lr' in sample_training_config
        assert 'pretrain_epochs' in sample_training_config
        assert 'finetune_epochs' in sample_training_config
        assert 'loss_function' in sample_training_config
        assert 'random_seed' in sample_training_config
        
        # Check values
        assert sample_training_config['pretrain_lr'] == 1e-3
        assert sample_training_config['finetune_lr'] == 1e-4
        assert sample_training_config['random_seed'] == 42


class TestUtilityFixtures:
    """Test utility fixtures."""
    
    def test_cifar10_class_names(self, cifar10_class_names):
        """Verify CIFAR-10 class names fixture."""
        assert len(cifar10_class_names) == 10
        assert 'airplane' in cifar10_class_names
        assert 'automobile' in cifar10_class_names
        assert 'bird' in cifar10_class_names
    
    def test_set_random_seed(self, set_random_seed):
        """Verify random seed fixture."""
        # Test reproducibility
        set_random_seed(42)
        result1 = torch.randn(10)
        
        set_random_seed(42)
        result2 = torch.randn(10)
        
        assert torch.allclose(result1, result2)
    
    def test_device(self, device):
        """Verify device fixture."""
        assert device in ['cpu', 'cuda']
        
        # Test tensor creation on device
        x = torch.randn(10).to(device)
        assert str(x.device).startswith(device)


class TestHypothesisConfiguration:
    """Test Hypothesis configuration."""
    
    def test_hypothesis_profile_loaded(self):
        """Verify Hypothesis profile is loaded."""
        from hypothesis import settings
        
        # Get current profile
        profile = settings.default
        
        # Check that max_examples is set (either 20 or 100)
        assert profile.max_examples in [20, 100]
        
        # Check that deadline is None
        assert profile.deadline is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
