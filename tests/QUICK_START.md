# Testing Quick Start Guide

## TL;DR

```bash
# Run all tests
pytest tests/ -v

# Run fast (dev mode)
export HYPOTHESIS_PROFILE=dev
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test type
pytest -m property -v    # Property-based tests
pytest -m unit -v        # Unit tests
pytest -m integration -v # Integration tests
```

## Common Commands

### Development Workflow

```bash
# 1. Run tests for component you're working on
pytest -m data_pipeline -v
pytest -m model -v
pytest -m loss -v

# 2. Run with coverage to see what's missing
pytest tests/unit_tests/ --cov=src --cov-report=term-missing

# 3. Run specific test file
pytest tests/unit_tests/test_model_architecture.py -v

# 4. Run specific test function
pytest tests/unit_tests/test_model_architecture.py::test_backbone_output_shape -v
```

### Before Committing

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Check if coverage meets target (90%+)
pytest tests/ --cov=src --cov-fail-under=90
```

### CI/CD Pipeline

```bash
# Property tests (thorough)
pytest tests/property_tests/ -v --hypothesis-profile=ci

# Unit tests with coverage
pytest tests/unit_tests/ -v --cov=src --cov-report=xml

# Integration tests
pytest tests/integration_tests/ -v
```

## Hypothesis Profiles

### Dev Profile (Fast - 20 examples)
```bash
export HYPOTHESIS_PROFILE=dev
pytest tests/property_tests/ -v
```

### CI Profile (Thorough - 100 examples)
```bash
export HYPOTHESIS_PROFILE=ci
pytest tests/property_tests/ -v
```

## Test Markers

Filter tests by marker:

```bash
pytest -m property -v          # Property-based tests
pytest -m unit -v              # Unit tests
pytest -m integration -v       # Integration tests
pytest -m slow -v              # Slow tests
pytest -m "not slow" -v        # Skip slow tests
pytest -m data_pipeline -v     # Data pipeline tests
pytest -m model -v             # Model tests
pytest -m loss -v              # Loss function tests
pytest -m training -v          # Training tests
pytest -m evaluation -v        # Evaluation tests
```

## Coverage Reports

### HTML Report (Interactive)
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Terminal Report
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### XML Report (for CI)
```bash
pytest tests/ --cov=src --cov-report=xml
```

## Writing Tests

### Property Test Template
```python
from hypothesis import given, strategies as st
import pytest

@pytest.mark.property
@given(st.lists(st.floats(min_value=0, max_value=1), min_size=10, max_size=10))
def test_my_property(input_data):
    """Property: What should always be true."""
    result = function_under_test(input_data)
    assert property_holds(result)
```

### Unit Test Template
```python
import pytest

@pytest.mark.unit
def test_specific_behavior(fixture_name):
    """Test specific behavior."""
    result = function_under_test(input_data)
    assert result == expected_value
```

### Using Fixtures
```python
def test_with_mock_data(mock_cifar10h_dataset):
    """Test using mock dataset fixture."""
    images = mock_cifar10h_dataset['images']
    assert images.shape == (100, 3, 32, 32)

def test_with_temp_dir(temp_output_dir):
    """Test using temporary directory."""
    output_file = temp_output_dir / "output.txt"
    output_file.write_text("test")
    assert output_file.exists()
```

## Troubleshooting

### Tests are slow
```bash
# Use dev profile
export HYPOTHESIS_PROFILE=dev

# Run specific tests
pytest tests/unit_tests/test_model.py -v

# Skip slow tests
pytest -m "not slow" -v
```

### Import errors
```bash
# Add project root to PYTHONPATH
export PYTHONPATH=.

# Or install in editable mode
pip install -e .
```

### Hypothesis found a counterexample
This is good! It found a bug. The output shows the failing input:
```
Falsifying example: test_my_property(
    input_data=[0.0, 0.0, 0.0, ...]
)
```
Fix your implementation to handle this case.

### Coverage is low
```bash
# See which lines are not covered
pytest tests/ --cov=src --cov-report=term-missing

# Focus on specific module
pytest tests/unit_tests/test_model.py --cov=src.model --cov-report=term-missing
```

## Available Fixtures

### Dataset Fixtures
- `mock_cifar10h_dataset` - Small mock dataset (100 samples)
- `mock_cifar10h_splits` - Train/val/test splits
- `sample_probability_distributions` - Various distribution types

### Model Fixtures
- `small_test_model` - Lightweight model for testing
- `pretrained_test_model` - Model with initialized weights

### Directory Fixtures
- `temp_output_dir` - Temporary output directory
- `temp_checkpoint_dir` - Temporary checkpoint directory
- `temp_data_dir` - Temporary data directory

### Configuration Fixtures
- `sample_data_pipeline_config` - Data pipeline config
- `sample_model_config` - Model config
- `sample_training_config` - Training config

### Utility Fixtures
- `cifar10_class_names` - CIFAR-10 class names
- `set_random_seed` - Set random seed function
- `device` - Device for testing (CPU/CUDA)

## Examples

### Test with Mock Dataset
```python
def test_data_loader(mock_cifar10h_dataset):
    images = mock_cifar10h_dataset['images']
    soft_labels = mock_cifar10h_dataset['soft_labels']
    
    assert images.shape == (100, 3, 32, 32)
    assert soft_labels.shape == (100, 10)
    assert torch.allclose(soft_labels.sum(dim=1), torch.ones(100))
```

### Test with Temporary Directory
```python
def test_save_checkpoint(small_test_model, temp_checkpoint_dir):
    checkpoint_path = temp_checkpoint_dir / "model.pth"
    torch.save(small_test_model.state_dict(), checkpoint_path)
    
    assert checkpoint_path.exists()
    loaded_state = torch.load(checkpoint_path)
    assert 'backbone.0.weight' in loaded_state
```

### Test with Random Seed
```python
def test_reproducibility(set_random_seed):
    set_random_seed(42)
    result1 = torch.randn(10)
    
    set_random_seed(42)
    result2 = torch.randn(10)
    
    assert torch.allclose(result1, result2)
```

## Need More Help?

See `tests/README.md` for comprehensive documentation.
