# CIFAR-10 Human Disagreement Predictor

A deep learning system that predicts human annotator disagreement on CIFAR-10 images. Rather than predicting a single hard class label, the system predicts the full probability distribution over labels that reflects how approximately 50 human annotators disagree about image classification.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Installation](#installation)
- [Dataset Download](#dataset-download)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
  - [Running Ablation Studies](#running-ablation-studies)
  - [End-to-End Pipeline](#end-to-end-pipeline)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Expected Performance](#expected-performance)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Overview

This project implements a modified ResNet-18 architecture with a two-stage training strategy:

1. **Pretraining** on CIFAR-10's 50,000 hard-labeled images to learn robust visual features
2. **Fine-tuning** on CIFAR-10H's 6,000 soft-labeled images to learn disagreement patterns

The system uses the CIFAR-10H dataset, which contains human annotation data showing how different annotators disagree on image classification. Instead of predicting a single class, the model predicts a probability distribution that reflects the natural disagreement among human annotators.

## Motivation

Traditional image classification systems predict a single "correct" label for each image. However, many images are genuinely ambiguous - a small blurry animal could reasonably be classified as either a cat or a dog. Human annotators naturally disagree on such images, and this disagreement contains valuable information about image ambiguity.

This project addresses three key questions:

1. **Can we predict human disagreement?** Rather than forcing a single classification, can a model learn to predict the distribution of human opinions?
2. **Does disagreement prediction improve robustness?** Models trained on soft labels (distributions) may be more robust to adversarial examples and distribution shift.
3. **What visual features drive disagreement?** Using explainability techniques like Grad-CAM, we can understand what image characteristics lead to human disagreement.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM
- 2GB+ disk space for datasets and checkpoints

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cifar10-disagreement-predictor
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- PyTorch 2.x with torchvision
- NumPy, Matplotlib, Pandas
- scikit-learn, scipy
- Hypothesis (for property-based testing)
- pytest (for testing)

## Dataset Download

### CIFAR-10

The standard CIFAR-10 dataset will be **automatically downloaded** when you run the data preparation script. No manual download required.

### CIFAR-10H

The CIFAR-10H dataset must be downloaded manually:

1. **Download from GitHub:**
   - Visit: https://github.com/jcpeterson/cifar-10h
   - Download the dataset files (or clone the repository)

2. **Extract and place files:**
   ```
   cifar-10h-1.0.0/
   └── data/
       ├── cifar10h-counts.npy
       └── cifar10h-probs.npy
   ```

3. **Verify files:**
   ```bash
   ls cifar-10h-1.0.0/data/
   # Should show: cifar10h-counts.npy  cifar10h-probs.npy
   ```

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
# Prepare data, train models, and evaluate
python run_pipeline.py
```

This will:
1. Download and prepare CIFAR-10 and CIFAR-10H datasets
2. Generate data visualizations
3. Pretrain a ResNet-18 on CIFAR-10 hard labels
4. Fine-tune three models with different loss functions (KL, JS, Custom)
5. Evaluate all models and generate comprehensive reports

### Data Preparation

Prepare datasets and generate visualizations:

```bash
python prepare_data.py
```

**Options:**
```bash
# Custom directories
python prepare_data.py --cifar10-dir ./my_data --cifar10h-dir ./my_cifar10h

# Custom output directory
python prepare_data.py --output-dir ./my_outputs

# Custom split sizes (must sum to 10000)
python prepare_data.py --train-size 5000 --val-size 2500 --test-size 2500

# Set random seed
python prepare_data.py --seed 42
```

**Outputs:**
- `outputs/data_visualizations/entropy_histogram.png` - Distribution of entropy values
- `outputs/data_visualizations/per_class_entropy.png` - Entropy by class
- `outputs/data_visualizations/example_grid.png` - Example images at different entropy levels
- `outputs/data_visualizations/data_pipeline_config.json` - Configuration
- `outputs/data_visualizations/split_info.json` - Dataset split information

### Training Models

Train disagreement prediction models:

```bash
# Train all three models (KL, JS, Custom) with default settings
python train.py
```

**Options:**
```bash
# Train only specific loss functions
python train.py --loss-functions kl js

# Custom hyperparameters
python train.py --pretrain-epochs 50 --finetune-epochs 30 --finetune-lr 5e-5

# Skip pretraining (use existing pretrained model)
python train.py --skip-pretrain --pretrained-path checkpoints/my_pretrained.pth

# Train on CPU
python train.py --device cpu

# Custom batch sizes
python train.py --pretrain-batch-size 64 --finetune-batch-size 32
```

**Training Process:**

1. **Pretraining Phase** (100 epochs by default):
   - Trains on 50,000 CIFAR-10 images with hard labels
   - Uses cross-entropy loss
   - Learning rate: 1e-3 with cosine annealing
   - Saves pretrained weights to `checkpoints/pretrained_resnet18_cifar10.pth`

2. **Fine-tuning Phase** (50 epochs max by default):
   - Trains on 6,000 CIFAR-10H images with soft labels
   - Uses KL/JS/Custom loss
   - Learning rate: 1e-4
   - Early stopping with patience=10
   - Saves best model to `checkpoints/finetuned_{loss}_best.pth`

**Outputs:**
- `checkpoints/pretrained_resnet18_cifar10.pth` - Pretrained model
- `checkpoints/finetuned_kl_best.pth` - KL-trained model
- `checkpoints/finetuned_js_best.pth` - JS-trained model
- `checkpoints/finetuned_custom_best.pth` - Custom loss-trained model
- `outputs/training_logs/pretrain_history.json` - Pretraining metrics
- `outputs/training_logs/finetune_{loss}_history.json` - Fine-tuning metrics

### Evaluating Models

Evaluate trained models on the test set:

```bash
# Evaluate all models in checkpoint directory
python evaluate.py

# Evaluate specific model
python evaluate.py --model-path checkpoints/finetuned_kl_best.pth

# Generate visualizations
python evaluate.py --generate-visualizations

# Evaluate robustness to corruptions
python evaluate.py --evaluate-robustness

# Specify number of failure cases to visualize
python evaluate.py --generate-visualizations --num-failure-cases 20
```

**Evaluation Metrics:**

1. **Distribution Matching:**
   - Mean KL Divergence (lower is better)
   - Mean JS Divergence (lower is better)
   - Mean Cosine Similarity (higher is better)

2. **Entropy Prediction Quality:**
   - Pearson Correlation (higher is better)
   - Spearman Correlation (higher is better)

3. **Ambiguity Identification:**
   - Precision@100, Precision@200, Precision@500

**Outputs:**
- `outputs/evaluation_results/evaluation_metrics.json` - All metrics
- `outputs/evaluation_results/per_class_performance.csv` - Per-class analysis
- `outputs/evaluation_results/model_comparison.csv` - Comparison across models
- `outputs/evaluation_results/gradcam_comparison.png` - Grad-CAM visualizations (if --generate-visualizations)
- `outputs/evaluation_results/failure_cases.png` - Top failure cases (if --generate-visualizations)
- `outputs/evaluation_results/entropy_correlation.png` - Entropy scatter plot (if --generate-visualizations)

### Running Ablation Studies

Run comprehensive ablation experiments:

```bash
# Run all ablation studies
python run_ablations.py

# Run specific studies
python run_ablations.py --studies loss initialization

# Custom epochs for faster experimentation
python run_ablations.py --pretrain-epochs 20 --finetune-epochs 10
```

**Ablation Studies:**

1. **Loss Functions** (`loss`): Compares KL, JS, and custom entropy-regularized loss
2. **Initialization** (`initialization`): Compares random vs CIFAR-10 pretraining
3. **Training Strategy** (`training_strategy`): Compares two-stage vs single-stage training
4. **Architecture** (`architecture`): Compares single-layer vs two-layer MLP prediction head

**Outputs:**
- `outputs/ablation_studies/{study_name}_comparison.csv` - Comparison tables
- Model checkpoints in `checkpoints/ablations/`

### End-to-End Pipeline

Run the complete pipeline from data preparation through evaluation:

```bash
# Run complete pipeline with default settings
python run_pipeline.py

# Use custom configuration file
python run_pipeline.py --config my_config.json

# Run specific phases only
python run_pipeline.py --phases data train evaluate

# Quick test run with reduced epochs
python run_pipeline.py --pretrain-epochs 10 --finetune-epochs 5
```

**Configuration File Example** (`example_config.json`):
```json
{
  "cifar10_dir": "./data",
  "cifar10h_dir": "./cifar-10h-1.0.0/data",
  "output_dir": "./outputs",
  "checkpoint_dir": "./checkpoints",
  "pretrain_epochs": 100,
  "finetune_epochs": 50,
  "loss_functions": ["kl", "js", "custom"],
  "generate_visualizations": true,
  "evaluate_robustness": false,
  "device": "cuda",
  "seed": 42
}
```

**Outputs:**
- `outputs/pipeline_config.json` - Pipeline configuration
- `outputs/pipeline_report.md` - Comprehensive markdown report
- All outputs from individual phases

## Project Structure

```
.
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_pipeline.py         # Data loading, preprocessing, splitting
│   ├── model.py                 # Modified ResNet-18 + MLP head
│   ├── losses.py                # KL, JS, Custom loss functions
│   ├── training.py              # Two-stage training protocol
│   ├── evaluation.py            # Comprehensive evaluation metrics
│   ├── visualization.py         # Plotting and Grad-CAM
│   ├── output_manager.py        # Output organization
│   └── logging_config.py        # Logging configuration
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_data_pipeline.py   # Data pipeline tests
│   ├── test_model.py            # Model architecture tests
│   ├── test_losses.py           # Loss function tests
│   ├── test_training.py         # Training tests
│   ├── test_evaluation.py      # Evaluation tests
│   └── property_tests/          # Property-based tests
│
├── data/                         # CIFAR-10 dataset (auto-downloaded)
├── cifar-10h-1.0.0/             # CIFAR-10H dataset (manual download)
│   └── data/
│       ├── cifar10h-counts.npy
│       └── cifar10h-probs.npy
│
├── outputs/                      # Experiment outputs
│   ├── data_visualizations/     # Data exploration plots
│   ├── training_logs/           # Training history
│   ├── evaluation_results/      # Evaluation metrics
│   ├── ablation_studies/        # Ablation results
│   └── explainability/          # Grad-CAM and failure analysis
│
├── checkpoints/                  # Model checkpoints
│   ├── pretrained_resnet18_cifar10.pth
│   ├── finetuned_kl_best.pth
│   ├── finetuned_js_best.pth
│   └── finetuned_custom_best.pth
│
├── prepare_data.py              # Data preparation script
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── run_ablations.py             # Ablation studies script
├── run_pipeline.py              # End-to-end pipeline script
│
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── README.md                    # This file
├── SCRIPTS_README.md            # Detailed script documentation
└── example_config.json          # Example configuration file
```

## Key Features

### Model Architecture
- **Modified ResNet-18 Backbone**: Adapted for 32×32 CIFAR-10 images (replaces 7×7 conv with 3×3, removes max pooling)
- **MLP Prediction Head**: Two-layer MLP (512→256→10) with ReLU and Softmax
- **Parameter Count**: ~11.13M parameters

### Loss Functions
1. **KL Divergence**: `KL(p || q)` - Asymmetric divergence measure
2. **JS Divergence**: `0.5*KL(p||m) + 0.5*KL(q||m)` - Symmetric, bounded divergence
3. **Custom Entropy-Regularized**: `KL(p||q) + λ|H(p)-H(q)|` - Explicitly penalizes entropy mismatch

### Training Strategy
- **Two-Stage Training**: Pretrain on 50k hard labels, fine-tune on 6k soft labels
- **Data Augmentation**: RandomHorizontalFlip + RandomCrop during training
- **Early Stopping**: Patience=10 based on validation KL divergence
- **Reproducibility**: Fixed random seed (42) for all operations

### Evaluation Metrics
- **Distribution Matching**: KL divergence, JS divergence, cosine similarity
- **Entropy Correlation**: Pearson and Spearman correlation
- **Precision@K**: Overlap in top-K ambiguous images (K=100, 200, 500)
- **Per-Class Analysis**: Metrics broken down by CIFAR-10 class
- **Robustness Testing**: Gaussian noise, blur, contrast reduction at 3 severity levels

### Explainability
- **Grad-CAM Visualization**: Attention maps showing what the model focuses on
- **Failure Case Analysis**: Identifies and visualizes worst predictions
- **Manual Categorization**: Interactive interface for categorizing disagreement sources

### Testing
- **Property-Based Testing**: 15 correctness properties validated with Hypothesis
- **Unit Tests**: Comprehensive coverage of all modules
- **Integration Tests**: End-to-end pipeline testing
- **Target Coverage**: 90%+ for core modules

## Expected Performance

Based on the CIFAR-10H test set (2,000 images):

| Metric | Target | Description |
|--------|--------|-------------|
| Mean KL Divergence | < 0.5 | Lower is better - measures distribution mismatch |
| Pearson Correlation (entropy) | > 0.7 | Higher is better - measures entropy prediction accuracy |
| Precision@100 | > 0.6 | Higher is better - identifies top 100 ambiguous images |
| Mean JS Divergence | < 0.3 | Lower is better - symmetric divergence measure |
| Mean Cosine Similarity | > 0.85 | Higher is better - distribution similarity |

**Training Time** (on NVIDIA RTX 3090):
- Pretraining: ~30 minutes (100 epochs)
- Fine-tuning: ~5 minutes per model (50 epochs max with early stopping)
- Total: ~45 minutes for all three models

**Inference Speed**:
- ~1000 images/second on GPU
- ~50 images/second on CPU

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Property-based tests only
pytest -m property

# Integration tests only
pytest -m integration
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run Specific Test Files

```bash
pytest tests/test_data_pipeline.py
pytest tests/test_model.py
pytest tests/test_losses.py
```

### Property-Based Testing

The project includes 15 property-based tests using Hypothesis:

1. Probability Distribution Normalization
2. Invalid Distribution Detection
3. Index-Based Alignment Preservation
4. Dataset Split Reproducibility
5. Dataset Split Disjointness
6. Paired Data Preservation During Splitting
7. Shannon Entropy Correctness
8. Entropy Numerical Stability
9. Entropy Bounds
10. Data Pipeline Configuration Round-Trip
11. Data Pipeline Configuration Error Reporting
12. Model Configuration Round-Trip
13. Model Configuration Error Reporting
14. Training Configuration Round-Trip
15. Training Configuration Error Reporting

## Troubleshooting

### Issue: CIFAR-10H not found

**Error:**
```
FileNotFoundError: CIFAR-10H counts file not found: ./cifar-10h-1.0.0/data/cifar10h-counts.npy
```

**Solution:**
1. Download CIFAR-10H from https://github.com/jcpeterson/cifar-10h
2. Extract files to `cifar-10h-1.0.0/data/`
3. Verify files exist:
   ```bash
   ls cifar-10h-1.0.0/data/
   # Should show: cifar10h-counts.npy  cifar10h-probs.npy
   ```

### Issue: Out of memory during training

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
python train.py --pretrain-batch-size 64 --finetune-batch-size 32

# Or use CPU (slower but no memory limit)
python train.py --device cpu
```

### Issue: Training too slow

**Solutions:**
```bash
# Use GPU if available
python train.py --device cuda

# Reduce epochs for testing
python train.py --pretrain-epochs 10 --finetune-epochs 5

# Use smaller dataset split for quick testing
python prepare_data.py --train-size 1000 --val-size 500 --test-size 500
```

### Issue: Import errors

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Checkpoint not found

**Error:**
```
CheckpointLoadError: Checkpoint file not found: checkpoints/pretrained_resnet18_cifar10.pth
```

**Solution:**
```bash
# Run pretraining first
python train.py

# Or specify correct checkpoint path
python evaluate.py --model-path path/to/your/checkpoint.pth
```

### Issue: Tests failing

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Run tests with verbose output to see details
pytest -v

# Run specific failing test
pytest tests/test_data_pipeline.py::test_specific_function -v
```

### Getting Help

If you encounter issues not covered here:

1. Check the error message carefully - it often includes helpful suggestions
2. Review the log files in `outputs/training_logs/`
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Verify dataset files are in the correct locations
5. Try running with `--log-level DEBUG` for more detailed output

## Citation

If you use this code or the CIFAR-10H dataset in your research, please cite:

```bibtex
@inproceedings{peterson2019human,
  title={Human uncertainty makes classification more robust},
  author={Peterson, Joshua C and Battleday, Ruairidh M and Griffiths, Thomas L and Russakovsky, Olga},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9617--9626},
  year={2019}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

The CIFAR-10H dataset is released under its own license. Please refer to the [CIFAR-10H repository](https://github.com/jcpeterson/cifar-10h) for dataset-specific licensing information.

## Acknowledgments

- CIFAR-10 dataset: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- CIFAR-10H dataset: Joshua C. Peterson, Ruairidh M. Battleday, Thomas L. Griffiths, and Olga Russakovsky
- ResNet architecture: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun

## Additional Resources

- **Detailed Script Documentation**: See `SCRIPTS_README.md` for comprehensive usage examples
- **Design Document**: See `.kiro/specs/cifar10-disagreement-predictor/design.md` for technical details
- **Requirements**: See `.kiro/specs/cifar10-disagreement-predictor/requirements.md` for formal specifications
- **Implementation Plan**: See `.kiro/specs/cifar10-disagreement-predictor/tasks.md` for development roadmap
# DNN_Final_Project
