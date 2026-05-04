"""
Unit tests for training module.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import json

from src.training import (
    set_seed,
    get_train_transform,
    get_test_transform,
    save_checkpoint,
    load_checkpoint,
    TrainingConfig
)
from src.model import DisagreementPredictor


class TestSeedManagement:
    """Test random seed management."""
    
    def test_set_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        # Set seed and generate random numbers
        set_seed(42)
        torch_rand_1 = torch.rand(5)
        
        # Set seed again and generate random numbers
        set_seed(42)
        torch_rand_2 = torch.rand(5)
        
        # Should be identical
        assert torch.allclose(torch_rand_1, torch_rand_2)
    
    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        torch_rand_1 = torch.rand(5)
        
        set_seed(123)
        torch_rand_2 = torch.rand(5)
        
        # Should be different
        assert not torch.allclose(torch_rand_1, torch_rand_2)


class TestDataAugmentation:
    """Test data augmentation transforms."""
    
    def test_train_transform_exists(self):
        """Test that training transform can be created."""
        transform = get_train_transform()
        assert transform is not None
    
    def test_test_transform_exists(self):
        """Test that test transform can be created."""
        transform = get_test_transform()
        assert transform is not None
    
    def test_train_transform_output_shape(self):
        """Test that training transform produces correct output shape."""
        transform = get_train_transform()
        # Create a dummy image (32x32x3)
        import numpy as np
        dummy_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Apply transform
        output = transform(dummy_image)
        
        # Should be (3, 32, 32) tensor
        assert output.shape == (3, 32, 32)
        assert isinstance(output, torch.Tensor)


class TestCheckpointManagement:
    """Test checkpoint save/load functionality."""
    
    def test_save_and_load_checkpoint(self):
        """Test that checkpoint can be saved and loaded."""
        model = DisagreementPredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        epoch = 10
        metrics = {'train_loss': 0.5, 'val_loss': 0.6}
        config = {'lr': 1e-4, 'batch_size': 64}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'checkpoint.pth')
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, metrics, filepath, config)
            
            # Create new model and load checkpoint
            new_model = DisagreementPredictor()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
            
            loaded_epoch, loaded_metrics, loaded_config = load_checkpoint(
                filepath, new_model, new_optimizer
            )
            
            # Verify loaded values
            assert loaded_epoch == epoch
            assert loaded_metrics == metrics
            assert loaded_config == config
    
    def test_load_checkpoint_file_not_found(self):
        """Test that loading non-existent checkpoint raises error."""
        from src.data_pipeline import CheckpointLoadError
        
        model = DisagreementPredictor()
        
        with pytest.raises(CheckpointLoadError):
            load_checkpoint('nonexistent.pth', model)
    
    def test_checkpoint_preserves_model_state(self):
        """Test that checkpoint preserves model weights."""
        model = DisagreementPredictor()
        
        # Get initial weights
        initial_weights = model.backbone.conv1.weight.clone()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'checkpoint.pth')
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, 0, {}, filepath)
            
            # Modify model weights
            with torch.no_grad():
                model.backbone.conv1.weight.fill_(0.0)
            
            # Load checkpoint
            load_checkpoint(filepath, model)
            
            # Weights should be restored
            assert torch.allclose(model.backbone.conv1.weight, initial_weights)


class TestTrainingConfig:
    """Test training configuration serialization."""
    
    def test_default_config_creation(self):
        """Test that default config can be created."""
        config = TrainingConfig()
        
        assert config.pretrain_epochs == 100
        assert config.finetune_epochs == 50
        assert config.pretrain_lr == 1e-3
        assert config.finetune_lr == 1e-4
        assert config.random_seed == 42
    
    def test_config_validation_valid(self):
        """Test that valid config passes validation."""
        config = TrainingConfig()
        config.validate()  # Should not raise
    
    def test_config_validation_invalid_epochs(self):
        """Test that invalid epochs fail validation."""
        config = TrainingConfig(pretrain_epochs=-1)
        
        with pytest.raises(ValueError, match="pretrain_epochs must be positive"):
            config.validate()
    
    def test_config_validation_invalid_lr(self):
        """Test that invalid learning rate fails validation."""
        config = TrainingConfig(pretrain_lr=-0.001)
        
        with pytest.raises(ValueError, match="pretrain_lr must be positive"):
            config.validate()
    
    def test_config_validation_invalid_loss_function(self):
        """Test that invalid loss function fails validation."""
        config = TrainingConfig(loss_function='invalid')
        
        with pytest.raises(ValueError, match="loss_function must be one of"):
            config.validate()
    
    def test_config_to_json(self):
        """Test that config can be serialized to JSON."""
        config = TrainingConfig(pretrain_epochs=50, random_seed=123)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'config.json')
            config.to_json(filepath)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert data['pretrain_epochs'] == 50
            assert data['random_seed'] == 123
    
    def test_config_from_json(self):
        """Test that config can be deserialized from JSON."""
        config = TrainingConfig(pretrain_epochs=50, random_seed=123)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'config.json')
            config.to_json(filepath)
            
            # Load config
            loaded_config = TrainingConfig.from_json(filepath)
            
            assert loaded_config.pretrain_epochs == 50
            assert loaded_config.random_seed == 123
    
    def test_config_round_trip(self):
        """Test that config survives round-trip serialization."""
        config = TrainingConfig(
            pretrain_epochs=75,
            finetune_epochs=30,
            pretrain_lr=5e-4,
            random_seed=999,
            loss_function='js'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'config.json')
            config.to_json(filepath)
            loaded_config = TrainingConfig.from_json(filepath)
            
            # All fields should match
            assert loaded_config.pretrain_epochs == config.pretrain_epochs
            assert loaded_config.finetune_epochs == config.finetune_epochs
            assert loaded_config.pretrain_lr == config.pretrain_lr
            assert loaded_config.random_seed == config.random_seed
            assert loaded_config.loss_function == config.loss_function
    
    def test_config_from_json_file_not_found(self):
        """Test that loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_json('nonexistent.json')
    
    def test_config_from_json_invalid_json(self):
        """Test that invalid JSON raises error."""
        from src.data_pipeline import ConfigParseError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'invalid.json')
            with open(filepath, 'w') as f:
                f.write('{ invalid json }')
            
            with pytest.raises(ConfigParseError, match="Invalid JSON"):
                TrainingConfig.from_json(filepath)
    
    def test_config_get_json_schema(self):
        """Test that JSON schema can be retrieved."""
        schema = TrainingConfig.get_json_schema()
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'pretrain_epochs' in schema['properties']
        assert 'loss_function' in schema['properties']


class TestTrainingFunctions:
    """Test training functions with mock data."""
    
    def test_pretrain_function_signature(self):
        """Test that pretrain function exists with correct signature."""
        from src.training import pretrain_on_hard_labels
        
        # Function should exist and be callable
        assert callable(pretrain_on_hard_labels)
    
    def test_finetune_function_signature(self):
        """Test that finetune function exists with correct signature."""
        from src.training import finetune_on_soft_labels
        
        # Function should exist and be callable
        assert callable(finetune_on_soft_labels)
    
    def test_train_all_models_function_exists(self):
        """Test that train_all_models function exists."""
        from src.training import train_all_models
        
        # Function should exist and be callable
        assert callable(train_all_models)
