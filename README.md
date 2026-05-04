# CIFAR-10 Human Disagreement Predictor

A deep learning system that predicts human annotator disagreement on CIFAR-10 images. Rather than predicting a single hard class label, the system predicts the full probability distribution over labels that reflects how approximately 50 human annotators disagree about image classification.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Model Architecture](#model-architecture)
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
- [Expected Results](#expected-results)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This project implements a modified ResNet-18 architecture trained in two stages:

1. **Pretraining** on CIFAR-10's 50,000 hard-labeled images to learn robust visual features
2. **Fine-tuning** on CIFAR-10H's 6,000 soft-labeled images to learn disagreement patterns

The system uses the [CIFAR-10H dataset](https://github.com/jcpeterson/cifar-10h), which records how ~50 human annotators label each of the 10,000 CIFAR-10 test images. Instead of predicting a single class, the model predicts a probability distribution that reflects the natural disagreement among human annotators.

---

## Motivation

Traditional image classifiers predict a single "correct" label. However, many images are genuinely ambiguous — a small blurry animal could reasonably be a cat or a dog. Human annotators naturally disagree on such images, and this disagreement contains valuable information about image ambiguity.

This project addresses three key questions:

1. **Can we predict human disagreement?** Can a model learn to predict the distribution of human opinions rather than forcing a single classification?
2. **Does disagreement prediction improve robustness?** Models trained on soft labels (distributions) may be more robust to adversarial examples and distribution shift.
3. **What visual features drive disagreement?** Using Grad-CAM, we can understand what image characteristics lead to human disagreement.

---

## Model Architecture

### Modified ResNet-18 Backbone

Standard ResNet-18 is designed for 224×224 ImageNet images. For 32×32 CIFAR-10 images, two modifications are made:

- **Replace initial 7×7 conv (stride 2)** with **3×3 conv (stride 1)** — preserves spatial resolution
- **Remove initial MaxPool layer** — prevents aggressive downsampling on small images

Feature dimensions through the backbone:

| Layer | Output Shape |
|-------|-------------|
| Input | (B, 3, 32, 32) |
| conv1 (3×3, s=1) | (B, 64, 32, 32) |
| layer1 | (B, 64, 32, 32) |
| layer2 | (B, 128, 16, 16) |
| layer3 | (B, 256, 8, 8) |
| layer4 | (B, 512, 4, 4) |
| avgpool | (B, 512) |

### MLP Prediction Head

```
512 → Linear(512, 256) → ReLU → Linear(256, 10) → Softmax
```

The softmax output is a valid probability distribution over 10 classes, representing the predicted annotator disagreement.

**Total parameters:** ~11.13M (ResNet-18 backbone ~11M + MLP head ~134K)

### Loss Functions

Three loss functions are supported for fine-tuning:

| Loss | Formula | Properties |
|------|---------|------------|
| **KL Divergence** | `KL(p ‖ q) = Σ p log(p/q)` | Asymmetric, unbounded |
| **Jensen-Shannon** | `0.5·KL(p‖m) + 0.5·KL(q‖m)` where `m = 0.5(p+q)` | Symmetric, bounded [0, log 2] |
| **Custom Entropy-Regularized** | `KL(p‖q) + λ·|H(p) − H(q)|`, λ=0.1 | Explicitly penalizes entropy mismatch |

All loss functions use ε=1e-7 for numerical stability.

### Two-Stage Training

| Stage | Dataset | Loss | LR | Batch | Epochs |
|-------|---------|------|----|-------|--------|
| Pretrain | CIFAR-10 train (50k) | Cross-entropy | 1e-3 | 128 | 100 |
| Fine-tune | CIFAR-10H train (6k) | KL / JS / Custom | 1e-4 | 64 | 50 (early stop) |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- ~4 GB RAM
- ~2 GB disk space for datasets and checkpoints

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cifar10-disagreement-predictor
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:
   - `torch>=2.0.0`, `torchvision>=0.15.0`
   - `numpy>=1.24.0`, `scipy>=1.10.0`, `scikit-learn>=1.2.0`
   - `matplotlib>=3.7.0`
   - `pytest>=7.3.0`, `hypothesis>=6.75.0`

---

## Dataset Download

### CIFAR-10

CIFAR-10 is **downloaded automatically** by torchvision when you run any script. No manual action required.

### CIFAR-10H

CIFAR-10H must be downloaded manually:

1. Visit https://github.com/jcpeterson/cifar-10h and download the dataset.

2. Place the files so the directory looks like:
   ```
   cifar-10h-1.0.0/
   └── data/
       ├── cifar10h-counts.npy
       └── cifar10h-probs.npy
   ```

3. Verify the files are present:
   ```bash
   ls cifar-10h-1.0.0/data/
   # cifar10h-counts.npy  cifar10h-probs.npy
   ```

---

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python run_pipeline.py
```

This will:
1. Download and prepare CIFAR-10 and CIFAR-10H
2. Generate data visualizations
3. Pretrain ResNet-18 on CIFAR-10 hard labels (100 epochs)
4. Fine-tune three models with KL, JS, and Custom loss (up to 50 epochs each)
5. Evaluate all models and generate a comprehensive report

---

### Data Preparation

```bash
python prepare_data.py [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cifar10-dir DIR` | `./data` | Directory to store/load CIFAR-10 |
| `--cifar10h-dir DIR` | `./cifar-10h-1.0.0/data` | Directory containing CIFAR-10H files |
| `--output-dir DIR` | `./outputs/data_visualizations` | Where to save visualizations |
| `--train-size N` | `6000` | Training split size |
| `--val-size N` | `2000` | Validation split size |
| `--test-size N` | `2000` | Test split size (must sum to 10000) |
| `--seed N` | `42` | Random seed |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

**Examples:**

```bash
# Default settings
python prepare_data.py

# Custom directories
python prepare_data.py --cifar10-dir ./my_data --cifar10h-dir ./my_cifar10h

# Custom split sizes
python prepare_data.py --train-size 5000 --val-size 2500 --test-size 2500
```

**Outputs:**
```
outputs/data_visualizations/
├── entropy_histogram.png       # Distribution of entropy values across all images
├── per_class_entropy.png       # Box plots of entropy per CIFAR-10 class
├── example_grid.png            # Low / medium / high entropy image examples
├── data_pipeline_config.json   # Serialized pipeline configuration
└── split_info.json             # Dataset split sizes and seed
```

---

### Training Models

```bash
python train.py [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cifar10-dir DIR` | `./data` | CIFAR-10 directory |
| `--cifar10h-dir DIR` | `./cifar-10h-1.0.0/data` | CIFAR-10H directory |
| `--checkpoint-dir DIR` | `./checkpoints` | Where to save model checkpoints |
| `--log-dir DIR` | `./outputs/training_logs` | Where to save training logs |
| `--loss-functions {kl,js,custom}` | `kl js custom` | Loss functions to train (space-separated) |
| `--pretrain-epochs N` | `100` | Pretraining epochs |
| `--pretrain-lr F` | `1e-3` | Pretraining learning rate |
| `--pretrain-batch-size N` | `128` | Pretraining batch size |
| `--finetune-epochs N` | `50` | Max fine-tuning epochs |
| `--finetune-lr F` | `1e-4` | Fine-tuning learning rate |
| `--finetune-batch-size N` | `64` | Fine-tuning batch size |
| `--weight-decay F` | `1e-4` | AdamW weight decay |
| `--early-stopping-patience N` | `10` | Early stopping patience (val KL) |
| `--lambda-weight F` | `0.1` | Entropy penalty weight for custom loss |
| `--skip-pretrain` | off | Skip pretraining, load existing weights |
| `--pretrained-path PATH` | `checkpoints/pretrained_resnet18_cifar10.pth` | Path to load pretrained weights |
| `--device {cuda,cpu}` | auto | Compute device |
| `--seed N` | `42` | Random seed |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

**Examples:**

```bash
# Train all three models with defaults
python train.py

# Train only KL and JS models
python train.py --loss-functions kl js

# Faster run for testing
python train.py --pretrain-epochs 10 --finetune-epochs 5

# Skip pretraining (use existing checkpoint)
python train.py --skip-pretrain --pretrained-path checkpoints/pretrained_resnet18_cifar10.pth

# Train on CPU
python train.py --device cpu

# Smaller batch sizes for limited GPU memory
python train.py --pretrain-batch-size 64 --finetune-batch-size 32
```

**Outputs:**
```
checkpoints/
├── pretrained_resnet18_cifar10.pth   # Pretrained backbone
├── finetuned_kl_best.pth             # Best KL-trained model
├── finetuned_js_best.pth             # Best JS-trained model
└── finetuned_custom_best.pth         # Best custom-loss model

outputs/training_logs/
├── pretrain_history.json             # Per-epoch train loss and accuracy
├── finetune_kl_history.json          # Per-epoch KL fine-tuning metrics
├── finetune_js_history.json
├── finetune_custom_history.json
└── training_config.json              # Serialized training configuration
```

---

### Evaluating Models

```bash
python evaluate.py [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | `None` | Evaluate a single checkpoint (if None, evaluates all in `--checkpoint-dir`) |
| `--checkpoint-dir DIR` | `./checkpoints` | Directory to scan for `finetuned_*_best.pth` files |
| `--cifar10-dir DIR` | `./data` | CIFAR-10 directory |
| `--cifar10h-dir DIR` | `./cifar-10h-1.0.0/data` | CIFAR-10H directory |
| `--output-dir DIR` | `./outputs/evaluation_results` | Where to save results |
| `--batch-size N` | `64` | Evaluation batch size |
| `--generate-visualizations` | off | Generate Grad-CAM and failure case plots |
| `--evaluate-robustness` | off | Evaluate robustness to image corruptions |
| `--num-failure-cases N` | `10` | Number of failure cases to visualize |
| `--device {cuda,cpu}` | auto | Compute device |
| `--seed N` | `42` | Random seed |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

**Examples:**

```bash
# Evaluate all models in checkpoints/
python evaluate.py

# Evaluate a single model
python evaluate.py --model-path checkpoints/finetuned_kl_best.pth

# Full evaluation with visualizations and robustness testing
python evaluate.py --generate-visualizations --evaluate-robustness

# Custom output directory
python evaluate.py --output-dir ./my_results
```

**Metrics reported:**

| Category | Metric | Direction |
|----------|--------|-----------|
| Distribution matching | Mean KL divergence | Lower is better |
| Distribution matching | Mean JS divergence | Lower is better |
| Distribution matching | Mean cosine similarity | Higher is better |
| Entropy prediction | Pearson correlation | Higher is better |
| Entropy prediction | Spearman correlation | Higher is better |
| Ambiguity ranking | Precision@100 | Higher is better |
| Ambiguity ranking | Precision@200 | Higher is better |
| Ambiguity ranking | Precision@500 | Higher is better |

**Outputs:**
```
outputs/evaluation_results/
├── model_comparison.csv                        # Side-by-side metric comparison
└── finetuned_kl_best/
    ├── evaluation_metrics.json                 # All metrics as JSON
    ├── per_class_performance.csv               # Per-class breakdown
    ├── gradcam_comparison.png                  # (--generate-visualizations)
    ├── failure_cases.png                       # (--generate-visualizations)
    ├── entropy_correlation.png                 # (--generate-visualizations)
    └── corruption_robustness.png               # (--evaluate-robustness)
```

---

### Running Ablation Studies

```bash
python run_ablations.py [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--studies {loss,initialization,training_strategy,architecture,all}` | all four | Which studies to run |
| `--cifar10-dir DIR` | `./data` | CIFAR-10 directory |
| `--cifar10h-dir DIR` | `./cifar-10h-1.0.0/data` | CIFAR-10H directory |
| `--output-dir DIR` | `./outputs/ablation_studies` | Where to save results |
| `--checkpoint-dir DIR` | `./checkpoints/ablations` | Where to save ablation checkpoints |
| `--pretrain-epochs N` | `100` | Pretraining epochs |
| `--finetune-epochs N` | `50` | Fine-tuning epochs |
| `--batch-size N` | `64` | Batch size |
| `--device {cuda,cpu}` | auto | Compute device |
| `--seed N` | `42` | Random seed |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

**Examples:**

```bash
# Run all ablation studies
python run_ablations.py

# Run only loss function and initialization studies
python run_ablations.py --studies loss initialization

# Faster run with fewer epochs
python run_ablations.py --pretrain-epochs 20 --finetune-epochs 10
```

**Studies available:**

| Study | What it compares |
|-------|-----------------|
| `loss` | KL vs JS vs Custom entropy-regularized loss |
| `initialization` | Random init vs CIFAR-10 pretrained |
| `training_strategy` | Two-stage (pretrain + finetune) vs single-stage (finetune only) |
| `architecture` | Single linear head (512→10) vs two-layer MLP (512→256→10) |

**Outputs:**
```
outputs/ablation_studies/
├── loss_functions/
├── initialization/
├── training_strategy/
└── architecture/
    └── *_comparison.csv    # Comparison tables for each study
```

---

### End-to-End Pipeline

```bash
python run_pipeline.py [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `None` | Load settings from a JSON config file |
| `--phases {data,train,evaluate,ablations,all}` | `data train evaluate` | Which phases to run |
| `--cifar10-dir DIR` | `./data` | CIFAR-10 directory |
| `--cifar10h-dir DIR` | `./cifar-10h-1.0.0/data` | CIFAR-10H directory |
| `--output-dir DIR` | `./outputs` | Base output directory |
| `--checkpoint-dir DIR` | `./checkpoints` | Checkpoint directory |
| `--pretrain-epochs N` | `100` | Pretraining epochs |
| `--finetune-epochs N` | `50` | Fine-tuning epochs |
| `--loss-functions {kl,js,custom}` | `kl js custom` | Loss functions to train |
| `--generate-visualizations` | on | Generate evaluation visualizations |
| `--evaluate-robustness` | off | Include robustness evaluation |
| `--device {cuda,cpu}` | auto | Compute device |
| `--seed N` | `42` | Random seed |
| `--generate-report` | on | Generate a markdown summary report |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

**Examples:**

```bash
# Full pipeline with defaults
python run_pipeline.py

# Use a JSON config file
python run_pipeline.py --config example_config.json

# Run only training and evaluation (skip data prep)
python run_pipeline.py --phases train evaluate

# Include ablation studies
python run_pipeline.py --phases all

# Quick smoke test
python run_pipeline.py --pretrain-epochs 5 --finetune-epochs 3
```

**Configuration file** (`example_config.json`):
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
```
outputs/
├── data_visualizations/        # Entropy plots and example grids
├── training_logs/              # Per-epoch training metrics
├── evaluation_results/         # Metrics, visualizations, comparisons
├── ablation_studies/           # (if --phases includes ablations)
├── pipeline_config.json        # Saved pipeline configuration
└── pipeline_report.md          # Comprehensive markdown report
```

---

## Project Structure

```
.
├── src/                          # Core modules
│   ├── __init__.py
│   ├── data_pipeline.py         # Data loading, alignment, splitting, entropy
│   ├── model.py                 # Modified ResNet-18 + MLP head
│   ├── losses.py                # KL, JS, and custom loss functions
│   ├── training.py              # Two-stage training protocol
│   ├── evaluation.py            # Metrics and ablation comparisons
│   ├── visualization.py         # Plots and Grad-CAM
│   ├── output_manager.py        # Output directory management
│   └── logging_config.py        # Logging setup
│
├── tests/                        # Test suite
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_data_pipeline.py
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_output_manager.py
│   ├── property_tests/          # Hypothesis property-based tests
│   ├── unit_tests/
│   └── integration_tests/
│
├── cifar-10h-1.0.0/             # CIFAR-10H dataset (manual download)
│   └── data/
│       ├── cifar10h-counts.npy  # Raw annotator counts (10000, 10)
│       └── cifar10h-probs.npy   # Probability distributions (10000, 10)
│
├── data/                         # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/                  # Saved model weights
│   ├── pretrained_resnet18_cifar10.pth
│   ├── finetuned_kl_best.pth
│   ├── finetuned_js_best.pth
│   └── finetuned_custom_best.pth
│
├── outputs/                      # All generated outputs
│   ├── data_visualizations/
│   ├── training_logs/
│   ├── evaluation_results/
│   ├── ablation_studies/
│   └── explainability/
│
├── prepare_data.py              # Data preparation script
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── run_ablations.py             # Ablation studies script
├── run_pipeline.py              # End-to-end pipeline script
│
├── requirements.txt
├── pytest.ini
├── example_config.json
└── README.md
```

---

## Expected Results

Evaluated on the CIFAR-10H test split (2,000 images):

| Metric | Target | Description |
|--------|--------|-------------|
| Mean KL Divergence | **< 0.5** | Distribution mismatch (lower is better) |
| Pearson r (entropy) | **> 0.7** | Entropy prediction accuracy (higher is better) |
| Precision@100 | **> 0.6** | Top-100 ambiguous image overlap (higher is better) |
| Mean JS Divergence | < 0.3 | Symmetric divergence (lower is better) |
| Mean Cosine Similarity | > 0.85 | Distribution similarity (higher is better) |

**Training time** (NVIDIA RTX 3090):
- Pretraining: ~30 minutes (100 epochs, 50k images)
- Fine-tuning: ~5 minutes per model (early stopping typically triggers before 50 epochs)
- Total for all three models: ~45 minutes

**Inference speed:**
- GPU: ~1,000 images/second
- CPU: ~50 images/second

---

## Testing

### Run all tests

```bash
pytest
```

### Run by category

```bash
pytest -m unit          # Unit tests only
pytest -m property      # Property-based tests only
pytest -m integration   # Integration tests only
```

### Run with coverage

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to view the report
```

### Run a specific test file

```bash
pytest tests/test_data_pipeline.py -v
pytest tests/test_losses.py -v
pytest tests/test_model.py -v
```

### Property-based tests

The project includes 15 property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/):

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

---

## Troubleshooting

### CIFAR-10H files not found

**Error:**
```
FileNotFoundError: CIFAR-10H counts file not found: ./cifar-10h-1.0.0/data/cifar10h-counts.npy
```

**Fix:** Download CIFAR-10H from https://github.com/jcpeterson/cifar-10h and place the `.npy` files in `cifar-10h-1.0.0/data/`. See [Dataset Download](#dataset-download).

---

### CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Fix:** Reduce batch sizes:
```bash
python train.py --pretrain-batch-size 64 --finetune-batch-size 32
```
Or fall back to CPU:
```bash
python train.py --device cpu
```

---

### Training is slow

**Fix:** Ensure you are using a GPU:
```bash
python train.py --device cuda
```
For a quick smoke test, reduce epochs:
```bash
python train.py --pretrain-epochs 5 --finetune-epochs 3
```

---

### Import errors / missing modules

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Fix:** Activate your virtual environment and reinstall dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### Checkpoint not found during evaluation

**Error:**
```
CheckpointLoadError: Checkpoint file not found: checkpoints/pretrained_resnet18_cifar10.pth
```

**Fix:** Run training first, or point to the correct checkpoint:
```bash
python train.py
# or
python evaluate.py --model-path path/to/your/checkpoint.pth
```

---

### NaN or Inf loss during training

**Error:**
```
NumericalInstabilityError: NaN detected in loss
```

**Fix:** This can happen with very high learning rates or corrupted data. Try:
```bash
python train.py --pretrain-lr 5e-4 --finetune-lr 5e-5
```

---

### Tests failing

**Fix:** Ensure all dependencies are installed and run with verbose output:
```bash
pip install -r requirements.txt
pytest -v tests/test_data_pipeline.py
```

For more detail on a specific failure:
```bash
pytest tests/test_losses.py::test_kl_identical_distributions -v --tb=long
```

---

### Getting more diagnostic output

Add `--log-level DEBUG` to any script for detailed per-batch logging:
```bash
python train.py --log-level DEBUG
python evaluate.py --log-level DEBUG
```

---

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

**CIFAR-10 dataset:**
- Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton

**ResNet architecture:**
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun

---

## Additional Resources

- `SCRIPTS_README.md` — Detailed script documentation and advanced usage examples
- `.kiro/specs/cifar10-disagreement-predictor/design.md` — Full technical design document
- `.kiro/specs/cifar10-disagreement-predictor/requirements.md` — Formal requirements specification
- `example_config.json` — Example pipeline configuration file
- `tests/README.md` — Testing infrastructure documentation
