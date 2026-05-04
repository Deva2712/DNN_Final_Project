# Implementation Plan: CIFAR-10 Human Disagreement Predictor

## Overview

This implementation plan breaks down the CIFAR-10 Human Disagreement Predictor project into discrete, actionable coding tasks. The system predicts probability distributions over class labels that reflect human annotator disagreement, using a modified ResNet-18 architecture with two-stage training (pretraining on hard labels, fine-tuning on soft labels).

The implementation follows 7 phases: Data Pipeline, Model Architecture, Loss Functions, Training Protocol, Evaluation, Robustness & Explainability, and Testing & Documentation.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `data/`, `outputs/`, `checkpoints/`
  - Create `requirements.txt` with PyTorch, torchvision, NumPy, Matplotlib, scikit-learn, scipy, hypothesis, pytest
  - Create `src/__init__.py` and module files: `data_pipeline.py`, `model.py`, `losses.py`, `training.py`, `evaluation.py`, `visualization.py`
  - Set up logging configuration
  - _Requirements: 28.1, 28.2, 28.3, 28.4, 28.5_

- [x] 2. Implement Phase 1: Data Pipeline
  - [x] 2.1 Implement CIFAR-10 and CIFAR-10H data loading
    - Download CIFAR-10 using `torchvision.datasets.CIFAR10`
    - Load CIFAR-10H counts from `cifar-10h-1.0.0/data/cifar10h-counts.npy`
    - Load CIFAR-10H probabilities from `cifar-10h-1.0.0/data/cifar10h-probs.npy`
    - Verify dataset sizes (CIFAR-10: 50k train + 10k test, CIFAR-10H: 10k images)
    - Implement error handling for missing files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_
  
  - [x] 2.2 Write property test for probability distribution normalization
    - **Property 1: Probability Distribution Normalization**
    - **Validates: Requirements 2.1, 2.2**
    - Use Hypothesis to generate random count arrays
    - Verify normalized distributions sum to 1.0 within epsilon=1e-7
  
  - [x] 2.3 Implement soft label computation and validation
    - Compute probability distributions by normalizing annotator counts
    - Validate all distributions sum to 1.0 within epsilon=1e-7
    - Raise ValidationError with image index for invalid distributions
    - Verify each soft label vector has exactly 10 values
    - _Requirements: 2.1, 2.2, 2.3, 2.5_
  
  - [ ]* 2.4 Write property test for invalid distribution detection
    - **Property 2: Invalid Distribution Detection**
    - **Validates: Requirement 2.3**
    - Generate invalid distributions (sum < 0.99 or sum > 1.01)
    - Verify ValidationError is raised with correct index
  
  - [x] 2.5 Implement dataset alignment
    - Align CIFAR-10H images with CIFAR-10 test set by index
    - Create list of (image, soft_label, hard_label) tuples
    - Verify alignment preserves index correspondence
    - _Requirements: 2.4_
  
  - [ ]* 2.6 Write property test for index-based alignment preservation
    - **Property 3: Index-Based Alignment Preservation**
    - **Validates: Requirement 2.4**
    - Generate mock datasets with known index mappings
    - Verify correspondence is preserved after alignment
  
  - [x] 2.7 Implement dataset splitting with fixed random seed
    - Split CIFAR-10H into 6000 train / 2000 val / 2000 test using seed=42
    - Use sklearn.model_selection.train_test_split
    - Verify no overlap between splits
    - Preserve image-label mappings during splitting
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ]* 2.8 Write property tests for dataset splitting
    - **Property 4: Dataset Split Reproducibility**
    - **Validates: Requirement 3.2**
    - Run split twice with seed=42, verify identical index sets
    - **Property 5: Dataset Split Disjointness**
    - **Validates: Requirement 3.3**
    - Verify train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅
    - **Property 6: Paired Data Preservation During Splitting**
    - **Validates: Requirement 3.4**
    - Verify image-label pairs remain intact in assigned splits
  
  - [x] 2.9 Implement Shannon entropy computation
    - Compute H(p) = -Σ p(y) * log₂(p(y)) for all soft labels
    - Use epsilon=1e-7 for numerical stability
    - Verify entropy values are in range [0, 3.32] bits
    - Store entropy values for train/val/test splits
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [ ]* 2.10 Write property tests for entropy computation
    - **Property 7: Shannon Entropy Correctness**
    - **Validates: Requirement 4.1**
    - Verify computed entropy equals -Σ p(y) * log₂(p(y))
    - **Property 8: Entropy Numerical Stability**
    - **Validates: Requirement 4.2**
    - Test distributions with zeros, verify finite values (no NaN/Inf)
    - **Property 9: Entropy Bounds**
    - **Validates: Requirement 4.3**
    - Verify 0 ≤ H(p) ≤ 3.32 for all distributions
  
  - [x] 2.11 Implement custom PyTorch Dataset class
    - Create `CIFAR10HDataset` class with `__init__`, `__getitem__`, `__len__`
    - Return (image, soft_label, hard_label, entropy) tuples
    - Support optional transforms for augmentation
    - _Requirements: 3.4, 4.4_
  
  - [x] 2.12 Implement data visualization functions
    - Create histogram of entropy distribution across all images
    - Create per-class entropy box plots
    - Create example grid showing low/medium/high entropy images with distributions
    - Save all visualizations to `outputs/data_visualizations/`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 2.13 Implement configuration serialization for data pipeline
    - Create `DataPipelineConfig` dataclass with all parameters
    - Implement `to_json()` and `from_json()` methods
    - Define JSON schema for validation
    - _Requirements: 32.1, 32.2, 32.5_
  
  - [ ]* 2.14 Write property tests for data pipeline configuration
    - **Property 10: Data Pipeline Configuration Round-Trip**
    - **Validates: Requirement 32.3**
    - Generate random configs, verify parse(serialize(config)) == config
    - **Property 11: Data Pipeline Configuration Error Reporting**
    - **Validates: Requirement 32.4**
    - Generate invalid configs, verify descriptive errors

- [x] 3. Checkpoint - Verify data pipeline
  - Run data loading and verify 10,000 CIFAR-10H images loaded
  - Verify train/val/test split sizes: 6000/2000/2000
  - Check entropy histogram shows reasonable distribution
  - Ensure all tests pass, ask the user if questions arise

- [x] 4. Implement Phase 2: Model Architecture
  - [x] 4.1 Implement modified ResNet-18 backbone for 32×32 images
    - Load ResNet-18 from torchvision
    - Replace 7×7 conv (stride 2) with 3×3 conv (stride 1)
    - Remove initial max pooling layer
    - Remove final fully connected layer
    - Verify output shape: (batch_size, 512) for 32×32 input
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 4.2 Implement MLP prediction head
    - Create `DisagreementPredictionHead` class
    - Implement 512 → 256 → 10 architecture with ReLU
    - Apply softmax to output logits
    - Verify output distributions sum to 1.0
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  
  - [x] 4.3 Implement complete DisagreementPredictor model
    - Combine backbone and prediction head
    - Implement `forward()` method
    - Implement `get_features()` method for analysis
    - Verify end-to-end output shape: (batch_size, 10)
    - _Requirements: 6.5, 7.1, 7.6_
  
  - [ ]* 4.4 Write unit tests for model architecture
    - Test backbone output shape for 32×32 input
    - Test prediction head outputs valid probability distributions
    - Test complete model forward pass
    - Test feature extraction method
  
  - [x] 4.5 Implement model configuration serialization
    - Create `ModelConfig` dataclass
    - Implement `to_json()` and `from_json()` methods
    - Define JSON schema for validation
    - _Requirements: 33.1, 33.2, 33.5_
  
  - [ ]* 4.6 Write property tests for model configuration
    - **Property 12: Model Configuration Round-Trip**
    - **Validates: Requirement 33.3**
    - Generate random configs, verify round-trip equality
    - **Property 13: Model Configuration Error Reporting**
    - **Validates: Requirement 33.4**
    - Generate invalid configs, verify descriptive errors

- [x] 5. Implement Phase 3: Loss Functions
  - [x] 5.1 Implement KL divergence loss function
    - Compute KL(p || q) = Σ p(y) * log(p(y) / q(y))
    - Add epsilon=1e-7 for numerical stability
    - Normalize distributions after adding epsilon
    - Return mean loss across batch
    - _Requirements: 9.1, 9.2, 9.3, 27.2_
  
  - [x] 5.2 Implement Jensen-Shannon divergence loss function
    - Compute mixture distribution m = 0.5 * (p + q)
    - Compute JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    - Add epsilon=1e-7 for numerical stability
    - Return mean loss across batch
    - _Requirements: 10.1, 10.2, 10.3, 27.3_
  
  - [x] 5.3 Implement custom entropy-regularized loss function
    - Compute KL divergence term
    - Compute Shannon entropy for both distributions
    - Compute L_custom = KL(p || q) + λ|H(p) - H(q)| with λ=0.1
    - Return combined loss
    - _Requirements: 11.1, 11.2, 11.3, 27.4_
  
  - [ ]* 5.4 Write unit tests for loss functions
    - Test KL(p || p) ≈ 0 for identical distributions
    - Test JS divergence symmetry: JS(p || q) = JS(q || p)
    - Test custom loss > KL loss when entropies differ
    - Test numerical stability with zero probabilities
    - Test loss functions don't produce NaN or Inf

- [x] 6. Implement Phase 4: Training Protocol
  - [x] 6.1 Implement random seed management
    - Create `set_seed()` function setting seeds for random, numpy, torch, cuda
    - Set `torch.backends.cudnn.deterministic = True`
    - Call at start of training with seed=42
    - _Requirements: 26.1, 26.2, 26.3, 26.4, 26.5_
  
  - [x] 6.2 Implement data augmentation transforms
    - Create training transform: RandomHorizontalFlip + RandomCrop(32, padding=4) + Normalize
    - Create test transform: Normalize only
    - Use CIFAR-10 normalization statistics
    - _Requirements: 12.4, 12.5, 12.6_
  
  - [x] 6.3 Implement pretraining on CIFAR-10 hard labels
    - Load CIFAR-10 training set (50,000 images)
    - Use cross-entropy loss
    - Use AdamW optimizer with lr=1e-3, weight_decay=1e-4
    - Use batch size 128
    - Train for 100 epochs with cosine annealing schedule
    - Log training loss and accuracy per epoch
    - Save pretrained weights
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 12.1, 12.2, 12.3, 13.1_
  
  - [x] 6.4 Implement fine-tuning on CIFAR-10H soft labels
    - Load pretrained weights
    - Use AdamW optimizer with lr=1e-4, weight_decay=1e-4
    - Use batch size 64
    - Train for up to 50 epochs
    - Implement early stopping with patience=10 based on validation KL divergence
    - Log train loss, val loss, val KL, val JS per epoch
    - Save best model checkpoint
    - _Requirements: 8.4, 9.4, 9.5, 10.4, 10.5, 11.4, 11.5, 12.1, 12.2, 12.3, 13.2, 13.3, 13.4, 13.5, 13.6_
  
  - [x] 6.5 Train three models with different loss functions
    - Train model with KL divergence loss
    - Train model with JS divergence loss
    - Train model with custom entropy-regularized loss
    - Save all three models with descriptive filenames
    - _Requirements: 9.4, 10.4, 11.4_
  
  - [x] 6.6 Implement checkpoint management
    - Save model state dict, optimizer state dict, metrics, config
    - Use naming convention: `finetuned_{loss_name}_best.pth`
    - Implement checkpoint loading function
    - _Requirements: 30.1, 30.2, 30.3, 30.4, 30.5_
  
  - [x] 6.7 Implement training configuration serialization
    - Create `TrainingConfig` dataclass
    - Implement `to_json()` and `from_json()` methods
    - Define JSON schema for validation
    - _Requirements: 34.1, 34.2, 34.5_
  
  - [ ]* 6.8 Write property tests for training configuration
    - **Property 14: Training Configuration Round-Trip**
    - **Validates: Requirement 34.3**
    - Generate random configs, verify round-trip equality
    - **Property 15: Training Configuration Error Reporting**
    - **Validates: Requirement 34.4**
    - Generate invalid configs, verify descriptive errors
  
  - [ ]* 6.9 Write unit tests for training components
    - Test seed setting produces reproducible results
    - Test augmentation transforms work correctly
    - Test checkpoint save/load preserves model state
    - Test early stopping triggers correctly

- [x] 7. Checkpoint - Verify training pipeline
  - Verify pretrained model achieves >70% accuracy on CIFAR-10
  - Verify fine-tuned models complete training without errors
  - Check training logs show decreasing validation loss
  - Ensure all tests pass, ask the user if questions arise

- [x] 8. Implement Phase 5: Evaluation Metrics
  - [x] 8.1 Implement distribution matching metrics
    - Compute KL divergence for each test sample
    - Compute JS divergence for each test sample
    - Compute cosine similarity for each test sample
    - Return mean and std for all metrics
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_
  
  - [x] 8.2 Implement entropy prediction quality metrics
    - Compute true entropy from annotator distributions
    - Compute predicted entropy from model distributions
    - Compute Pearson correlation between true and predicted entropy
    - Compute Spearman correlation between true and predicted entropy
    - Generate scatter plot with correlation coefficients
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_
  
  - [x] 8.3 Implement Precision@K evaluation
    - Rank test images by true entropy (descending)
    - Rank test images by predicted entropy (descending)
    - Compute overlap for K=100, 200, 500
    - Return Precision@K for all K values
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_
  
  - [x] 8.4 Implement comprehensive evaluation function
    - Combine all metrics into single evaluation function
    - Return `EvaluationMetrics` dataclass with all results
    - Export metrics to JSON file
    - _Requirements: 14.1-14.6, 15.1-15.5, 16.1-16.5, 31.1, 31.2, 31.3_
  
  - [ ]* 8.5 Write unit tests for evaluation metrics
    - Test KL divergence computation
    - Test JS divergence computation
    - Test cosine similarity computation
    - Test entropy correlation computation
    - Test Precision@K computation

- [x] 9. Implement Phase 5: Ablation Studies
  - [x] 9.1 Implement loss function comparison
    - Evaluate all three models (KL, JS, Custom) on test set
    - Generate comparison table with all metrics
    - Highlight best-performing loss for each metric
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_
  
  - [x] 9.2 Implement backbone initialization ablation
    - Train model with random initialization (no pretraining)
    - Train model with CIFAR-10 pretraining (baseline)
    - Compare both strategies across all metrics
    - Generate comparison table
    - _Requirements: 18.1, 18.2, 18.4, 18.5_
  
  - [x] 9.3 Implement training strategy ablation
    - Train model with two-stage training (pretrain + finetune)
    - Train model with single-stage training (finetune only)
    - Compare both strategies across all metrics
    - Generate comparison table
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5_
  
  - [x] 9.4 Implement prediction head architecture ablation
    - Train model with single linear layer (512→10)
    - Train model with two-layer MLP (512→256→10)
    - Compare both architectures across all metrics
    - Generate comparison table
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5_
  
  - [x] 9.5 Implement per-class performance analysis
    - Compute mean KL, JS, Pearson correlation for each of 10 classes
    - Generate per-class performance table
    - Identify best and worst performing classes
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5_
  
  - [ ]* 9.6 Write unit tests for ablation studies
    - Test comparison table generation
    - Test per-class metric computation
    - Test best/worst class identification

- [x] 10. Implement Phase 6: Robustness Testing
  - [x] 10.1 Implement image corruption functions
    - Implement Gaussian noise corruption (severity 1, 3, 5)
    - Implement Gaussian blur corruption (severity 1, 3, 5)
    - Implement contrast reduction corruption (severity 1, 3, 5)
    - _Requirements: 21.1, 21.2, 21.3_
  
  - [x] 10.2 Implement corruption robustness evaluation
    - Apply each corruption at each severity to test images
    - Measure entropy change compared to clean images
    - Generate plot showing entropy change vs severity
    - _Requirements: 21.4, 21.5_
  
  - [ ]* 10.3 Write unit tests for corruption functions
    - Test corruption functions preserve image shape
    - Test corruption functions produce valid pixel values [0, 1]
    - Test severity levels produce increasing corruption

- [x] 11. Implement Phase 6: Explainability
  - [x] 11.1 Implement Grad-CAM visualization
    - Create `GradCAM` class with forward/backward hooks
    - Implement `generate_cam()` method for heatmap generation
    - Target final convolutional layer (layer4)
    - _Requirements: 23.3, 23.4_
  
  - [x] 11.2 Implement Grad-CAM comparison visualization
    - Select low-entropy images (high agreement)
    - Select high-entropy images (high disagreement)
    - Generate Grad-CAM heatmaps for selected images
    - Create visualization grid comparing attention patterns
    - _Requirements: 23.1, 23.2, 23.5_
  
  - [x] 11.3 Implement failure case analysis
    - Identify images with highest KL divergence (worst predictions)
    - Display original image, true distribution, predicted distribution
    - Display true and predicted entropy values
    - Generate visualization grid for top 10 failure cases
    - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.6_
  
  - [x] 11.4 Implement manual disagreement categorization interface
    - Select 20-30 highest entropy images
    - Display each image with annotator distribution
    - Provide categories: ambiguous identity, poor quality, multi-object, boundary case, other
    - Allow manual categorization
    - Generate summary report with category frequencies
    - _Requirements: 25.1, 25.2, 25.3, 25.4, 25.5_
  
  - [ ]* 11.5 Write unit tests for explainability components
    - Test Grad-CAM produces heatmaps of correct shape
    - Test failure case identification
    - Test categorization summary generation

- [x] 12. Checkpoint - Verify evaluation and analysis
  - Verify all three models evaluated successfully
  - Check ablation study results are reasonable
  - Review Grad-CAM visualizations for sanity
  - Ensure all tests pass, ask the user if questions arise

- [x] 13. Implement Phase 7: Testing Infrastructure
  - [x] 13.1 Set up pytest configuration
    - Create `pytest.ini` with test discovery settings
    - Create `conftest.py` with shared fixtures
    - Configure Hypothesis settings for CI and dev profiles
    - _Requirements: Testing Strategy section_
  
  - [x] 13.2 Create shared test fixtures
    - Fixture for mock CIFAR-10H dataset
    - Fixture for small test model
    - Fixture for temporary output directories
    - Fixture for sample probability distributions
  
  - [ ]* 13.3 Implement all property-based tests
    - Implement all 15 properties defined in design document
    - Use Hypothesis for test case generation
    - Configure 100 iterations for CI, 20 for dev
    - Ensure all properties pass
  
  - [ ]* 13.4 Implement comprehensive unit tests
    - Test all data pipeline functions
    - Test all model components
    - Test all loss functions
    - Test all training utilities
    - Test all evaluation metrics
    - Test all visualization functions
    - Target 90%+ code coverage
  
  - [ ]* 13.5 Implement integration tests
    - Test end-to-end training pipeline
    - Test end-to-end evaluation pipeline
    - Test checkpoint save/load workflow
    - Test configuration serialization workflow

- [x] 14. Implement error handling and logging
  - [x] 14.1 Implement custom exception classes
    - Create `ValidationError`, `DataShapeError`, `ConfigParseError`
    - Create `NumericalInstabilityError`, `CheckpointLoadError`
    - Add descriptive error messages
    - _Requirements: Error Handling section_
  
  - [x] 14.2 Add error handling to data pipeline
    - Handle missing CIFAR-10H files with FileNotFoundError
    - Validate data shapes and raise DataShapeError
    - Validate soft labels and raise ValidationError
    - _Requirements: 2.3, 27.1, 27.2, 27.3, 27.4_
  
  - [x] 14.3 Add numerical stability checks to training
    - Check for NaN/Inf loss values during training
    - Raise NumericalInstabilityError with diagnostic info
    - _Requirements: 27.5_
  
  - [x] 14.4 Set up comprehensive logging
    - Configure logging with file and console handlers
    - Add DEBUG logs for batch-level details
    - Add INFO logs for epoch-level progress
    - Add WARNING logs for potential issues
    - Add ERROR/CRITICAL logs for failures
    - _Requirements: Logging Strategy section_
  
  - [ ]* 14.5 Write unit tests for error handling
    - Test custom exceptions are raised correctly
    - Test error messages are descriptive
    - Test logging captures appropriate information

- [x] 15. Implement output management
  - [x] 15.1 Create output directory structure
    - Create `outputs/data_visualizations/`
    - Create `outputs/training_logs/`
    - Create `outputs/evaluation_results/`
    - Create `outputs/ablation_studies/`
    - Create `outputs/explainability/`
    - Create `checkpoints/`
    - _Requirements: 29.1, 29.7_
  
  - [x] 15.2 Implement visualization saving
    - Save all plots with descriptive filenames
    - Include timestamps in filenames
    - Organize by experiment type
    - _Requirements: 29.2, 29.3, 29.4, 29.5, 29.6, 29.7_
  
  - [x] 15.3 Implement metrics export
    - Export all metrics to JSON files
    - Export comparison tables to CSV files
    - Include metadata (timestamp, config, seed)
    - _Requirements: 31.1, 31.2, 31.3, 31.4, 31.5_

- [x] 16. Create main execution scripts
  - [x] 16.1 Create data preparation script
    - Script to download and prepare all datasets
    - Generate data visualizations
    - Save dataset splits and configurations
  
  - [x] 16.2 Create training script
    - Script to train all three models (KL, JS, Custom)
    - Support command-line arguments for hyperparameters
    - Save checkpoints and training logs
  
  - [x] 16.3 Create evaluation script
    - Script to evaluate trained models on test set
    - Generate all metrics and visualizations
    - Export results to JSON/CSV
  
  - [x] 16.4 Create ablation study script
    - Script to run all ablation experiments
    - Generate comparison tables
    - Export results
  
  - [x] 16.5 Create end-to-end pipeline script
    - Script to run complete pipeline from data prep to evaluation
    - Support configuration via JSON files
    - Generate comprehensive report

- [ ] 17. Write project documentation
  - [ ] 17.1 Create comprehensive README.md
    - Project overview and motivation
    - Installation instructions
    - Dataset download instructions
    - Usage examples for all scripts
    - Expected results and performance metrics
    - Troubleshooting guide
  
  - [ ] 17.2 Create API documentation
    - Document all public functions and classes
    - Include parameter descriptions and return types
    - Add usage examples
    - Generate documentation with Sphinx or similar
  
  - [ ] 17.3 Create experiment report template
    - Template for documenting experiment results
    - Sections for each phase of evaluation
    - Placeholders for tables and figures
    - Analysis and interpretation guidelines

- [ ] 18. Final checkpoint - Complete system verification
  - Run complete end-to-end pipeline
  - Verify all 15 property tests pass
  - Verify unit test coverage >90%
  - Verify all three models train successfully
  - Verify evaluation metrics are computed correctly
  - Verify all visualizations are generated
  - Verify ablation studies complete successfully
  - Review README and documentation for completeness
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property-based tests use Hypothesis with 100 iterations in CI
- All random operations use seed=42 for reproducibility
- Target code coverage: 90%+ for core modules
- Expected total implementation time: 11-16 days
- Success criteria: Mean KL < 0.5, Pearson r > 0.7, Precision@100 > 0.6

## Property-Based Tests Summary

The design document defines 15 correctness properties to be tested:

1. **Probability Distribution Normalization** - Validates Requirements 2.1, 2.2
2. **Invalid Distribution Detection** - Validates Requirement 2.3
3. **Index-Based Alignment Preservation** - Validates Requirement 2.4
4. **Dataset Split Reproducibility** - Validates Requirement 3.2
5. **Dataset Split Disjointness** - Validates Requirement 3.3
6. **Paired Data Preservation During Splitting** - Validates Requirement 3.4
7. **Shannon Entropy Correctness** - Validates Requirement 4.1
8. **Entropy Numerical Stability** - Validates Requirement 4.2
9. **Entropy Bounds** - Validates Requirement 4.3
10. **Data Pipeline Configuration Round-Trip** - Validates Requirement 32.3
11. **Data Pipeline Configuration Error Reporting** - Validates Requirement 32.4
12. **Model Configuration Round-Trip** - Validates Requirement 33.3
13. **Model Configuration Error Reporting** - Validates Requirement 33.4
14. **Training Configuration Round-Trip** - Validates Requirement 34.3
15. **Training Configuration Error Reporting** - Validates Requirement 34.4

All property tests should be implemented using the Hypothesis library with appropriate test case generation strategies.
