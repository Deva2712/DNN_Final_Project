"""
Verification script for Phase 2: Model Architecture implementation.

This script verifies that all components of Task 4 are working correctly:
- Task 4.1: Modified ResNet-18 backbone
- Task 4.2: MLP prediction head
- Task 4.3: Complete DisagreementPredictor model
- Task 4.5: Model configuration serialization
"""

import torch
import tempfile
import os
from src.model import (
    create_modified_resnet18,
    DisagreementPredictionHead,
    DisagreementPredictor,
    ModelConfig
)


def verify_task_4_1():
    """Verify Task 4.1: Modified ResNet-18 backbone for 32×32 images."""
    print("\n" + "="*80)
    print("Task 4.1: Verify Modified ResNet-18 Backbone")
    print("="*80)
    
    backbone = create_modified_resnet18()
    
    # Verify modifications
    print("✓ Initial conv layer: 3×3 with stride 1")
    assert backbone.conv1.kernel_size == (3, 3)
    assert backbone.conv1.stride == (1, 1)
    
    print("✓ Max pooling layer removed (replaced with Identity)")
    assert isinstance(backbone.maxpool, torch.nn.Identity)
    
    print("✓ Final FC layer removed (replaced with Identity)")
    assert isinstance(backbone.fc, torch.nn.Identity)
    
    # Verify output shape
    backbone.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        features = backbone(x)
    
    print(f"✓ Output shape for 32×32 input: {features.shape} (expected: torch.Size([8, 512]))")
    assert features.shape == (8, 512)
    
    print("\n✅ Task 4.1 PASSED: Modified ResNet-18 backbone working correctly")


def verify_task_4_2():
    """Verify Task 4.2: MLP prediction head."""
    print("\n" + "="*80)
    print("Task 4.2: Verify MLP Prediction Head")
    print("="*80)
    
    head = DisagreementPredictionHead(input_dim=512, hidden_dim=256, num_classes=10)
    
    # Verify architecture
    print("✓ Architecture: 512 → 256 → 10")
    assert head.fc1.in_features == 512
    assert head.fc1.out_features == 256
    assert head.fc2.in_features == 256
    assert head.fc2.out_features == 10
    
    print("✓ ReLU activation present")
    assert isinstance(head.relu, torch.nn.ReLU)
    
    print("✓ Softmax activation present")
    assert isinstance(head.softmax, torch.nn.Softmax)
    
    # Verify output is valid probability distribution
    head.eval()
    features = torch.randn(16, 512)
    with torch.no_grad():
        output = head(features)
    
    sums = output.sum(dim=1)
    print(f"✓ Output distributions sum to 1.0: {sums[0]:.6f} (all within tolerance)")
    assert torch.allclose(sums, torch.ones(16), atol=1e-6)
    
    print(f"✓ Output shape: {output.shape} (expected: torch.Size([16, 10]))")
    assert output.shape == (16, 10)
    
    print("\n✅ Task 4.2 PASSED: MLP prediction head working correctly")


def verify_task_4_3():
    """Verify Task 4.3: Complete DisagreementPredictor model."""
    print("\n" + "="*80)
    print("Task 4.3: Verify Complete DisagreementPredictor Model")
    print("="*80)
    
    model = DisagreementPredictor()
    
    # Verify components
    print("✓ Backbone component present")
    assert hasattr(model, 'backbone')
    
    print("✓ Prediction head component present")
    assert hasattr(model, 'head')
    assert isinstance(model.head, DisagreementPredictionHead)
    
    # Verify forward() method
    model.eval()
    x = torch.randn(16, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ forward() output shape: {output.shape} (expected: torch.Size([16, 10]))")
    assert output.shape == (16, 10)
    
    sums = output.sum(dim=1)
    print(f"✓ forward() outputs valid probability distributions (sum={sums[0]:.6f})")
    assert torch.allclose(sums, torch.ones(16), atol=1e-6)
    
    # Verify get_features() method
    with torch.no_grad():
        features = model.get_features(x)
    
    print(f"✓ get_features() output shape: {features.shape} (expected: torch.Size([16, 512]))")
    assert features.shape == (16, 512)
    
    print("\n✅ Task 4.3 PASSED: Complete DisagreementPredictor model working correctly")


def verify_task_4_5():
    """Verify Task 4.5: Model configuration serialization."""
    print("\n" + "="*80)
    print("Task 4.5: Verify Model Configuration Serialization")
    print("="*80)
    
    # Create ModelConfig
    config = ModelConfig(
        backbone_type='resnet18',
        input_dim=512,
        hidden_dim=256,
        num_classes=10,
        pretrained=False
    )
    
    print("✓ ModelConfig dataclass created")
    
    # Test to_json()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'model_config.json')
        config.to_json(filepath)
        
        print(f"✓ to_json() method works (saved to {filepath})")
        assert os.path.exists(filepath)
        
        # Test from_json()
        loaded_config = ModelConfig.from_json(filepath)
        
        print("✓ from_json() method works")
        assert loaded_config.backbone_type == config.backbone_type
        assert loaded_config.input_dim == config.input_dim
        assert loaded_config.hidden_dim == config.hidden_dim
        assert loaded_config.num_classes == config.num_classes
        assert loaded_config.pretrained == config.pretrained
        
        # Test round-trip
        filepath2 = os.path.join(tmpdir, 'model_config2.json')
        loaded_config.to_json(filepath2)
        config2 = ModelConfig.from_json(filepath2)
        
        print("✓ Round-trip serialization works (parse → serialize → parse)")
        assert config2.backbone_type == config.backbone_type
        assert config2.input_dim == config.input_dim
        assert config2.hidden_dim == config.hidden_dim
        assert config2.num_classes == config.num_classes
        assert config2.pretrained == config.pretrained
    
    # Test JSON schema
    schema = ModelConfig.get_json_schema()
    print("✓ JSON schema defined")
    assert 'properties' in schema
    assert 'backbone_type' in schema['properties']
    
    print("\n✅ Task 4.5 PASSED: Model configuration serialization working correctly")


def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("PHASE 2: MODEL ARCHITECTURE - VERIFICATION SCRIPT")
    print("="*80)
    print("\nThis script verifies the implementation of Task 4 and all its subtasks:")
    print("  - Task 4.1: Modified ResNet-18 backbone for 32×32 images")
    print("  - Task 4.2: MLP prediction head")
    print("  - Task 4.3: Complete DisagreementPredictor model")
    print("  - Task 4.5: Model configuration serialization")
    
    try:
        verify_task_4_1()
        verify_task_4_2()
        verify_task_4_3()
        verify_task_4_5()
        
        print("\n" + "="*80)
        print("✅ ALL TASKS PASSED - PHASE 2 IMPLEMENTATION COMPLETE")
        print("="*80)
        print("\nSummary:")
        print("  ✓ Modified ResNet-18 backbone outputs 512-dim features for 32×32 input")
        print("  ✓ MLP prediction head outputs valid probability distributions")
        print("  ✓ Complete model forward pass produces (batch_size, 10) output")
        print("  ✓ Feature extraction method works correctly")
        print("  ✓ Model configuration round-trip serialization works")
        print("\nAll requirements validated:")
        print("  - Requirements 6.1, 6.2, 6.3, 6.4, 6.5 (Backbone)")
        print("  - Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6 (Prediction Head)")
        print("  - Requirements 33.1, 33.2, 33.5 (Configuration)")
        print("\n" + "="*80)
        
    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == '__main__':
    main()
