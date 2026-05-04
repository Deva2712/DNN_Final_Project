"""
Unit tests for model architecture module.

Tests the modified ResNet-18 backbone, MLP prediction head, complete model,
and configuration serialization.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import json
from src.model import (
    DisagreementPredictionHead,
    create_modified_resnet18,
    DisagreementPredictor,
    ModelConfig
)


class TestDisagreementPredictionHead:
    """Tests for the MLP prediction head."""
    
    def test_initialization(self):
        """Test that prediction head initializes with correct architecture."""
        head = DisagreementPredictionHead(input_dim=512, hidden_dim=256, num_classes=10)
        
        # Check layers exist
        assert isinstance(head.fc1, nn.Linear)
        assert isinstance(head.fc2, nn.Linear)
        assert isinstance(head.relu, nn.ReLU)
        assert isinstance(head.softmax, nn.Softmax)
        
        # Check dimensions
        assert head.fc1.in_features == 512
        assert head.fc1.out_features == 256
        assert head.fc2.in_features == 256
        assert head.fc2.out_features == 10
    
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        head = DisagreementPredictionHead(input_dim=512, hidden_dim=256, num_classes=10)
        
        # Create dummy input
        batch_size = 16
        features = torch.randn(batch_size, 512)
        
        # Forward pass
        output = head(features)
        
        # Check output shape
        assert output.shape == (batch_size, 10)
    
    def test_output_is_probability_distribution(self):
        """Test that output is a valid probability distribution (sums to 1.0)."""
        head = DisagreementPredictionHead(input_dim=512, hidden_dim=256, num_classes=10)
        
        # Create dummy input
        batch_size = 16
        features = torch.randn(batch_size, 512)
        
        # Forward pass
        output = head(features)
        
        # Check that each row sums to 1.0
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-6)
    
    def test_output_values_in_valid_range(self):
        """Test that all output values are in [0, 1]."""
        head = DisagreementPredictionHead(input_dim=512, hidden_dim=256, num_classes=10)
        
        # Create dummy input
        features = torch.randn(16, 512)
        
        # Forward pass
        output = head(features)
        
        # Check all values are in [0, 1]
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)
    
    def test_custom_dimensions(self):
        """Test that head works with custom dimensions."""
        head = DisagreementPredictionHead(input_dim=1024, hidden_dim=512, num_classes=20)
        
        features = torch.randn(8, 1024)
        output = head(features)
        
        assert output.shape == (8, 20)
        assert torch.allclose(output.sum(dim=1), torch.ones(8), atol=1e-6)


class TestModifiedResNet18:
    """Tests for the modified ResNet-18 backbone."""
    
    def test_backbone_creation(self):
        """Test that modified ResNet-18 is created successfully."""
        backbone = create_modified_resnet18()
        
        # Check that it's a ResNet model
        assert hasattr(backbone, 'conv1')
        assert hasattr(backbone, 'layer1')
        assert hasattr(backbone, 'layer2')
        assert hasattr(backbone, 'layer3')
        assert hasattr(backbone, 'layer4')
    
    def test_initial_conv_modification(self):
        """Test that initial 7x7 conv is replaced with 3x3 conv."""
        backbone = create_modified_resnet18()
        
        # Check conv1 is 3x3 with stride 1
        assert backbone.conv1.kernel_size == (3, 3)
        assert backbone.conv1.stride == (1, 1)
        assert backbone.conv1.in_channels == 3
        assert backbone.conv1.out_channels == 64
    
    def test_maxpool_removed(self):
        """Test that max pooling layer is removed."""
        backbone = create_modified_resnet18()
        
        # Check maxpool is Identity
        assert isinstance(backbone.maxpool, nn.Identity)
    
    def test_fc_removed(self):
        """Test that final fully connected layer is removed."""
        backbone = create_modified_resnet18()
        
        # Check fc is Identity
        assert isinstance(backbone.fc, nn.Identity)
    
    def test_output_shape_for_32x32_input(self):
        """Test that backbone outputs 512-dim features for 32x32 input."""
        backbone = create_modified_resnet18()
        backbone.eval()
        
        # Create dummy 32x32 input
        batch_size = 8
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Forward pass
        with torch.no_grad():
            features = backbone(x)
        
        # Check output shape is (batch_size, 512)
        assert features.shape == (batch_size, 512)
    
    def test_different_batch_sizes(self):
        """Test that backbone works with different batch sizes."""
        backbone = create_modified_resnet18()
        backbone.eval()
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 3, 32, 32)
            with torch.no_grad():
                features = backbone(x)
            assert features.shape == (batch_size, 512)


class TestDisagreementPredictor:
    """Tests for the complete DisagreementPredictor model."""
    
    def test_model_initialization(self):
        """Test that complete model initializes successfully."""
        model = DisagreementPredictor()
        
        # Check components exist
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'head')
        assert isinstance(model.head, DisagreementPredictionHead)
    
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        model = DisagreementPredictor()
        model.eval()
        
        # Create dummy input
        batch_size = 16
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 10)
    
    def test_output_is_probability_distribution(self):
        """Test that model output is a valid probability distribution."""
        model = DisagreementPredictor()
        model.eval()
        
        # Create dummy input
        batch_size = 16
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check that each row sums to 1.0
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-6)
    
    def test_get_features_method(self):
        """Test that get_features() extracts 512-dim features correctly."""
        model = DisagreementPredictor()
        model.eval()
        
        # Create dummy input
        batch_size = 8
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Extract features
        with torch.no_grad():
            features = model.get_features(x)
        
        # Check feature shape
        assert features.shape == (batch_size, 512)
    
    def test_end_to_end_forward_pass(self):
        """Test complete end-to-end forward pass."""
        model = DisagreementPredictor()
        model.eval()
        
        # Create dummy input
        x = torch.randn(4, 3, 32, 32)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Verify output properties
        assert output.shape == (4, 10)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)
        assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-6)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        model = DisagreementPredictor()
        model.train()
        
        # Create dummy input and target
        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 10)
        target = target / target.sum(dim=1, keepdim=True)  # Normalize
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        loss = ((output - target) ** 2).sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert model.head.fc1.weight.grad is not None
        assert model.head.fc2.weight.grad is not None


class TestModelConfig:
    """Tests for ModelConfig serialization."""
    
    def test_default_initialization(self):
        """Test that ModelConfig initializes with default values."""
        config = ModelConfig()
        
        assert config.backbone_type == 'resnet18'
        assert config.input_dim == 512
        assert config.hidden_dim == 256
        assert config.num_classes == 10
        assert config.pretrained == False
    
    def test_custom_initialization(self):
        """Test that ModelConfig accepts custom values."""
        config = ModelConfig(
            backbone_type='resnet34',
            input_dim=1024,
            hidden_dim=512,
            num_classes=20,
            pretrained=True
        )
        
        assert config.backbone_type == 'resnet34'
        assert config.input_dim == 1024
        assert config.hidden_dim == 512
        assert config.num_classes == 20
        assert config.pretrained == True
    
    def test_to_json_creates_file(self):
        """Test that to_json() creates a valid JSON file."""
        config = ModelConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model_config.json')
            config.to_json(filepath)
            
            # Check file exists
            assert os.path.exists(filepath)
            
            # Check file contains valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert data['backbone_type'] == 'resnet18'
            assert data['input_dim'] == 512
            assert data['hidden_dim'] == 256
            assert data['num_classes'] == 10
            assert data['pretrained'] == False
    
    def test_from_json_loads_config(self):
        """Test that from_json() loads configuration correctly."""
        original_config = ModelConfig(
            backbone_type='resnet18',
            input_dim=512,
            hidden_dim=256,
            num_classes=10,
            pretrained=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model_config.json')
            original_config.to_json(filepath)
            
            # Load config
            loaded_config = ModelConfig.from_json(filepath)
            
            # Check all fields match
            assert loaded_config.backbone_type == original_config.backbone_type
            assert loaded_config.input_dim == original_config.input_dim
            assert loaded_config.hidden_dim == original_config.hidden_dim
            assert loaded_config.num_classes == original_config.num_classes
            assert loaded_config.pretrained == original_config.pretrained
    
    def test_round_trip_serialization(self):
        """Test that config survives round-trip serialization (parse -> serialize -> parse)."""
        original_config = ModelConfig(
            backbone_type='resnet34',
            input_dim=1024,
            hidden_dim=512,
            num_classes=20,
            pretrained=True
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model_config.json')
            
            # First serialization
            original_config.to_json(filepath)
            
            # First deserialization
            config1 = ModelConfig.from_json(filepath)
            
            # Second serialization
            filepath2 = os.path.join(tmpdir, 'model_config2.json')
            config1.to_json(filepath2)
            
            # Second deserialization
            config2 = ModelConfig.from_json(filepath2)
            
            # Check all fields match original
            assert config2.backbone_type == original_config.backbone_type
            assert config2.input_dim == original_config.input_dim
            assert config2.hidden_dim == original_config.hidden_dim
            assert config2.num_classes == original_config.num_classes
            assert config2.pretrained == original_config.pretrained
    
    def test_from_json_missing_file(self):
        """Test that from_json() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ModelConfig.from_json('/nonexistent/path/config.json')
    
    def test_from_json_invalid_json(self):
        """Test that from_json() raises ConfigParseError for invalid JSON."""
        from src.data_pipeline import ConfigParseError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'invalid.json')
            
            # Write invalid JSON
            with open(filepath, 'w') as f:
                f.write('{ invalid json }')
            
            with pytest.raises(ConfigParseError, match="Invalid JSON"):
                ModelConfig.from_json(filepath)
    
    def test_validate_invalid_backbone(self):
        """Test that validate() raises error for invalid backbone type."""
        config = ModelConfig(backbone_type='invalid_backbone')
        
        with pytest.raises(ValueError, match="Invalid backbone_type"):
            config.validate()
    
    def test_validate_negative_input_dim(self):
        """Test that validate() raises error for negative input_dim."""
        config = ModelConfig(input_dim=-1)
        
        with pytest.raises(ValueError, match="input_dim must be positive"):
            config.validate()
    
    def test_validate_negative_hidden_dim(self):
        """Test that validate() raises error for negative hidden_dim."""
        config = ModelConfig(hidden_dim=0)
        
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            config.validate()
    
    def test_validate_negative_num_classes(self):
        """Test that validate() raises error for negative num_classes."""
        config = ModelConfig(num_classes=-5)
        
        with pytest.raises(ValueError, match="num_classes must be positive"):
            config.validate()
    
    def test_validate_invalid_pretrained_type(self):
        """Test that validate() raises error for non-boolean pretrained."""
        # This test checks runtime validation, not type hints
        config = ModelConfig()
        config.pretrained = "yes"  # Invalid type
        
        with pytest.raises(ValueError, match="pretrained must be boolean"):
            config.validate()
    
    def test_get_json_schema(self):
        """Test that get_json_schema() returns valid schema."""
        schema = ModelConfig.get_json_schema()
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'backbone_type' in schema['properties']
        assert 'input_dim' in schema['properties']
        assert 'hidden_dim' in schema['properties']
        assert 'num_classes' in schema['properties']
        assert 'pretrained' in schema['properties']
        assert 'required' in schema


class TestModelIntegration:
    """Integration tests for model components working together."""
    
    def test_model_with_real_cifar_dimensions(self):
        """Test model with realistic CIFAR-10 batch."""
        model = DisagreementPredictor()
        model.eval()
        
        # Simulate a real CIFAR-10 batch
        batch_size = 64
        x = torch.randn(batch_size, 3, 32, 32)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_model_parameter_count(self):
        """Test that model has expected number of parameters."""
        model = DisagreementPredictor()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        head_params = sum(p.numel() for p in model.head.parameters())
        
        # ResNet-18 backbone should have ~11M parameters
        assert backbone_params > 10_000_000
        assert backbone_params < 12_000_000
        
        # MLP head should have 512*256 + 256*10 = 133,632 parameters (plus biases)
        expected_head_params = 512 * 256 + 256 + 256 * 10 + 10
        assert head_params == expected_head_params
        
        # Total should be sum of both
        assert total_params == backbone_params + head_params
    
    def test_model_can_be_saved_and_loaded(self):
        """Test that model state can be saved and loaded."""
        model = DisagreementPredictor()
        
        # Create dummy input
        x = torch.randn(4, 3, 32, 32)
        
        # Get initial output
        model.eval()
        with torch.no_grad():
            output1 = model(x)
        
        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.pth')
            torch.save(model.state_dict(), filepath)
            
            # Create new model and load weights
            model2 = DisagreementPredictor()
            model2.load_state_dict(torch.load(filepath))
            model2.eval()
            
            # Get output from loaded model
            with torch.no_grad():
                output2 = model2(x)
            
            # Outputs should be identical
            assert torch.allclose(output1, output2, atol=1e-6)
