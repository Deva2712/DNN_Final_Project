# Test Suite for CIFAR-10 Human Disagreement Predictor

This directory contains the comprehensive test suite for the CIFAR-10 Human Disagreement Predictor project.

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures and Hypothesis configuration
├── property_tests/                # Property-based tests using Hypothesis
│   ├── test_data_pipeline_properties.py
│   ├── test_entropy_properties.py
│   ├── test_config_properties.py
│   └── test_split_properties.py
├── unit_tests/                    # Unit tests for individual components
│   ├── test_model_architecture.py
│   ├── test_loss_functions.py
│   ├── test_training.py
│   └── test_evaluation.py
└── integration_tests/             # Integration tests for complete workflows
    ├── test_end_to_end.py
    └── test_ablations.py
```

## Test Types

### Property-Based Tests (PBT)

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/) to automatically generate test cases and validate universal properties. These tests ensure correctness across all possible inputs.

**Example properties:**
- Probability distributions always sum to 1.0
- Shannon entropy is always in range [0, 3.32] for 10 classes
- Configuration round-trip (serialize → deserialize) preserves values
- Dataset splits are disjoint and reproducible

**Running property tests:**
```bash
# Run with CI profile (100 examples per test)
pytest tests/property_tests/ -v --hypothesis-profile=ci

# Run with dev profile (20 examples per test, faster)
pytest tests/property_tests/ -v --hypothesis-profile=dev
```

### Unit Tests

Unit tests validate specific behaviors and edge cases for individual components.

**Example tests:**
- Model architecture outputs correct shapes
- Loss functions handle edge cases (identical distributions, zeros)
- Data loaders return correct batch sizes
- Entropy computation handles numerical stability

**Running unit tests:**
```bash
pytest tests/unit_tests/ -v
```

### Integration Tests

Integration tests validate complete workflows end-to-end.

**Example tests:**
- Complete training pipeline (data loading → pretraining → fine-tuning → evaluation)
- Checkpoint save/load workflow
- Configuration serialization workflow
- Ablation study execution

**Running integration tests:**
```bash
pytest tests/integration_tests/ -v
```

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Tests by Marker
```bash
# Run only property tests
pytest -m property -v

# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run tests for specific component
pytest -m data_pipeline -v
pytest -m model -v
pytest -m loss -v
```

### Run with Coverage
```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# Generate terminal coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate XML coverage report (for CI)
pytest tests/ --cov=src --cov-report=xml
```

### Run Specific Test File
```bash
pytest tests/property_tests/test_entropy_properties.py -v
```

### Run Specific Test Function
```bash
pytest tests/property_tests/test_entropy_properties.py::test_entropy_computation_correctness -v
```

## Hypothesis Configuration

The test suite uses two Hypothesis profiles:

### CI Profile (Default)
- **max_examples**: 100
- **deadline**: None
- **Use case**: Continuous integration, thorough testing

### Dev Profile
- **max_examples**: 20
- **deadline**: None
- **Use case**: Local development, faster feedback

**Switch profiles:**
```bash
# Use dev profile
export HYPOTHESIS_PROFILE=dev
pytest tests/property_tests/ -v

# Use CI profile (default)
export HYPOTHESIS_PROFILE=ci
pytest tests/property_tests/ -v
```

## Fixtures

The `conftest.py` file provides shared fixtures for all tests:

### Dataset Fixtures
- `mock_cifar10h_dataset`: Small mock dataset (100 samples)
- `mock_cifar10h_splits`: Train/val/test splits of mock dataset
- `sample_probability_distributions`: Various probability distributions for testing

### Model Fixtures
- `small_test_model`: Lightweight model for fast testing
- `pretrained_test_model`: Small model with initialized weights

### Directory Fixtures
- `temp_output_dir`: Temporary directory for test outputs
- `temp_checkpoint_dir`: Temporary directory for checkpoints
- `temp_data_dir`: Temporary directory for test data

### Configuration Fixtures
- `sample_data_pipeline_config`: Sample data pipeline configuration
- `sample_model_config`: Sample model configuration
- `sample_training_config`: Sample training configuration

### Utility Fixtures
- `cifar10_class_names`: CIFAR-10 class names
- `set_random_seed`: Function to set random seed
- `device`: Device for testing (CPU or CUDA)

## Coverage Goals

### Target Coverage
- **Property tests**: 100% coverage of universal properties (15 properties)
- **Unit tests**: 90%+ code coverage for core modules
- **Integration tests**: All major workflows covered

### Coverage by Module
- Data Pipeline: 95%+ (critical for data integrity)
- Model Architecture: 90%+ (well-defined structure)
- Loss Functions: 100% (mathematical correctness critical)
- Training Module: 85%+ (some paths hard to test)
- Evaluation Module: 90%+ (metrics computation critical)

## Continuous Integration

The test suite is designed to run in CI pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run property tests
  run: pytest tests/property_tests/ -v --hypothesis-profile=ci

- name: Run unit tests with coverage
  run: pytest tests/unit_tests/ -v --cov=src --cov-report=xml

- name: Run integration tests
  run: pytest tests/integration_tests/ -v
```

## Writing New Tests

### Property-Based Test Template
```python
from hypothesis import given, strategies as st
import pytest

@pytest.mark.property
@given(st.lists(st.floats(min_value=0, max_value=1), min_size=10, max_size=10))
def test_my_property(input_data):
    """
    Property: Description of what should always be true.
    """
    # Arrange
    processed_data = preprocess(input_data)
    
    # Act
    result = function_under_test(processed_data)
    
    # Assert
    assert property_holds(result), "Property violation message"
```

### Unit Test Template
```python
import pytest

@pytest.mark.unit
def test_specific_behavior(fixture_name):
    """Test specific behavior with concrete example."""
    # Arrange
    input_data = setup_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

### Integration Test Template
```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_workflow(temp_output_dir):
    """Test complete workflow from start to finish."""
    # Arrange
    setup_environment()
    
    # Act
    result = run_complete_workflow()
    
    # Assert
    assert workflow_succeeded(result)
    assert output_files_exist(temp_output_dir)
```

## Troubleshooting

### Tests Fail with "Hypothesis found a counterexample"
This means Hypothesis discovered an input that violates the property. The output shows the failing input. Fix the implementation or adjust the property definition.

### Tests are Slow
- Use the dev profile for faster local testing: `export HYPOTHESIS_PROFILE=dev`
- Run specific test files instead of the entire suite
- Use pytest's `-k` flag to run tests matching a pattern

### Coverage is Low
- Check which lines are not covered: `pytest --cov=src --cov-report=term-missing`
- Add unit tests for uncovered branches
- Some code paths (error handling, edge cases) may be hard to cover

### Import Errors
- Ensure the project root is in PYTHONPATH: `export PYTHONPATH=.`
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify pytest is installed: `pip install pytest hypothesis pytest-cov`

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
