# Main Execution Scripts - User Guide

This document provides comprehensive instructions for using the five main execution scripts that implement the complete CIFAR-10 Human Disagreement Predictor pipeline.

## Overview

The project provides five main scripts that can be run independently or as part of an end-to-end pipeline:

1. **`prepare_data.py`** - Data preparation and visualization
2. **`train.py`** - Model training (pretraining + fine-tuning)
3. **`evaluate.py`** - Model evaluation and metrics generation
4. **`run_ablations.py`** - Ablation studies
5. **`run_pipeline.py`** - End-to-end pipeline orchestration

## Prerequisites

Ensure you have:
- Python 3.8+
- All dependencies installed: `pip install -r requirements.txt`
- CIFAR-10H dataset downloaded to `./cifar-10h-1.0.0/data/`

## Script 1: Data Preparation (`prepare_data.py`)

Downloads and prepares CIFAR-10 and CIFAR-10H datasets, generates visualizations, and saves dataset splits.

### Basic Usage

```bash
# Run with default settings
python prepare_data.py

# Specify custom directories
python prepare_data.py --cifar10-dir ./my_data --cifar10h-dir ./my_cifar10h

# Custom output directory
python prepare_data.py --output-dir ./my_outputs

# Custom split sizes (must sum to 10000)
python prepare_data.py --train-size 5000 --val-size 2500 --test-size 2500
```

### Key Arguments

- `--cifar10-dir`: Directory for CIFAR-10 dataset (default: `./data`)
- `--cifar10h-dir`: Directory for CIFAR-10H dataset (default: `./cifar-10h-1.0.0/data`)
- `--output-dir`: Output directory for visualizations (default: `./outputs/data_visualizations`)
- `--train-size`: Number of training samples (default: 6000)
- `--val-size`: Number of validation samples (default: 2000)
- `--test-size`: Number of test samples (default: 2000)
- `--seed`: Random seed (default: 42)
- `--download` / `--no-download`: Download CIFAR-10 if not present

### Outputs

- `entropy_histogram.png` - Distribution of entropy values
- `per_class_entropy.png` - Entropy distribution by class
- `example_grid.png` - Example images at different entropy levels
- `data_pipeline_config.json` - Configuration file
- `split_info.json` - Dataset split information

## Script 2: Training (`train.py`)

Trains disagreement prediction models with different loss functions. Supports both pretraining and fine-tuning phases.

### Basic Usage

```bash
# Train all three models (KL, JS, Custom) with default settings
python train.py

# Train only specific loss functions
python train.py --loss-functions kl js

# Custom hyperparameters
python train.py --pretrain-epochs 50 --finetune-epochs 30 --finetune-lr 5e-5

# Skip pretraining (use existing pretrained model)
python train.py --skip-pretrain --pretrained-path checkpoints/my_pretrained.pth

# Train on CPU
python train.py --device cpu
```

### Key Arguments

**Data:**
- `--cifar10-dir`: CIFAR-10 directory (default: `./data`)
- `--cifar10h-dir`: CIFAR-10H directory (default: `./cifar-10h-1.0.0/data`)

**Output:**
- `--checkpoint-dir`: Checkpoint directory (default: `./checkpoints`)
- `--log-dir`: Training logs directory (default: `./outputs/training_logs`)

**Loss Functions:**
- `--loss-functions`: Loss functions to train with (choices: `kl`, `js`, `custom`, default: all three)

**Pretraining:**
- `--pretrain-epochs`: Number of pretraining epochs (default: 100)
- `--pretrain-lr`: Pretraining learning rate (default: 1e-3)
- `--pretrain-batch-size`: Pretraining batch size (default: 128)
- `--skip-pretrain`: Skip pretraining phase
- `--pretrained-path`: Path to pretrained weights (default: `checkpoints/pretrained_resnet18_cifar10.pth`)

**Fine-tuning:**
- `--finetune-epochs`: Maximum fine-tuning epochs (default: 50)
- `--finetune-lr`: Fine-tuning learning rate (default: 1e-4)
- `--finetune-batch-size`: Fine-tuning batch size (default: 64)
- `--weight-decay`: Weight decay for AdamW (default: 1e-4)
- `--early-stopping-patience`: Early stopping patience (default: 10)

**Custom Loss:**
- `--lambda-weight`: Weight for entropy penalty in custom loss (default: 0.1)

**Other:**
- `--device`: Device to use (`cuda` or `cpu`)
- `--seed`: Random seed (default: 42)

### Outputs

- `checkpoints/pretrained_resnet18_cifar10.pth` - Pretrained model
- `checkpoints/finetuned_{loss}_best.pth` - Fine-tuned models for each loss function
- `outputs/training_logs/pretrain_history.json` - Pretraining history
- `outputs/training_logs/finetune_{loss}_history.json` - Fine-tuning history for each loss
- `outputs/training_logs/training_config.json` - Training configuration

## Script 3: Evaluation (`evaluate.py`)

Evaluates trained models on the test set, generates comprehensive metrics and visualizations.

### Basic Usage

```bash
# Evaluate all models in checkpoint directory
python evaluate.py

# Evaluate specific model
python evaluate.py --model-path checkpoints/finetuned_kl_best.pth

# Custom output directory
python evaluate.py --output-dir ./my_evaluation_results

# Generate all visualizations
python evaluate.py --generate-visualizations

# Evaluate robustness to corruptions
python evaluate.py --evaluate-robustness

# Specify number of failure cases to visualize
python evaluate.py --generate-visualizations --num-failure-cases 20
```

### Key Arguments

**Model:**
- `--model-path`: Path to specific model (if None, evaluates all models in checkpoint-dir)
- `--checkpoint-dir`: Directory containing checkpoints (default: `./checkpoints`)

**Data:**
- `--cifar10-dir`: CIFAR-10 directory (default: `./data`)
- `--cifar10h-dir`: CIFAR-10H directory (default: `./cifar-10h-1.0.0/data`)

**Output:**
- `--output-dir`: Output directory (default: `./outputs/evaluation_results`)

**Evaluation Options:**
- `--batch-size`: Batch size (default: 64)
- `--generate-visualizations`: Generate Grad-CAM and failure case visualizations
- `--evaluate-robustness`: Evaluate robustness to image corruptions
- `--num-failure-cases`: Number of failure cases to visualize (default: 10)

**Other:**
- `--device`: Device to use (`cuda` or `cpu`)
- `--seed`: Random seed (default: 42)

### Outputs

For each model:
- `evaluation_metrics.json` - All evaluation metrics
- `per_class_performance.csv` - Per-class analysis
- `gradcam_comparison.png` - Grad-CAM attention patterns (if `--generate-visualizations`)
- `failure_cases.png` - Top failure cases (if `--generate-visualizations`)
- `entropy_correlation.png` - Entropy correlation scatter plot (if `--generate-visualizations`)
- `corruption_robustness.png` - Robustness plot (if `--evaluate-robustness`)

If multiple models evaluated:
- `model_comparison.csv` - Comparison table across all models

## Script 4: Ablation Studies (`run_ablations.py`)

Runs comprehensive ablation experiments to understand the impact of different design choices.

### Basic Usage

```bash
# Run all ablation studies
python run_ablations.py

# Run specific ablation studies
python run_ablations.py --studies loss initialization

# Custom epochs for faster experimentation
python run_ablations.py --pretrain-epochs 20 --finetune-epochs 10

# Run all studies
python run_ablations.py --studies all
```

### Key Arguments

**Ablation Studies:**
- `--studies`: Which studies to run (choices: `loss`, `initialization`, `training_strategy`, `architecture`, `all`)

**Data:**
- `--cifar10-dir`: CIFAR-10 directory (default: `./data`)
- `--cifar10h-dir`: CIFAR-10H directory (default: `./cifar-10h-1.0.0/data`)

**Output:**
- `--output-dir`: Output directory (default: `./outputs/ablation_studies`)
- `--checkpoint-dir`: Checkpoint directory (default: `./checkpoints/ablations`)

**Training:**
- `--pretrain-epochs`: Pretraining epochs (default: 100)
- `--finetune-epochs`: Fine-tuning epochs (default: 50)
- `--batch-size`: Batch size (default: 64)

**Other:**
- `--device`: Device to use (`cuda` or `cpu`)
- `--seed`: Random seed (default: 42)

### Ablation Studies

1. **Loss Functions** (`loss`): Compares KL, JS, and custom entropy-regularized loss
2. **Initialization** (`initialization`): Compares random vs CIFAR-10 pretraining
3. **Training Strategy** (`training_strategy`): Compares two-stage vs single-stage training
4. **Architecture** (`architecture`): Compares single-layer vs two-layer MLP prediction head

### Outputs

For each ablation study:
- `{study_name}_comparison.csv` - Comparison table
- Model checkpoints in `checkpoints/ablations/`

## Script 5: End-to-End Pipeline (`run_pipeline.py`)

Orchestrates the complete pipeline from data preparation through evaluation. Generates a comprehensive report.

### Basic Usage

```bash
# Run complete pipeline with default settings
python run_pipeline.py

# Use custom configuration file
python run_pipeline.py --config my_config.json

# Run specific phases only
python run_pipeline.py --phases data train evaluate

# Quick test run with reduced epochs
python run_pipeline.py --pretrain-epochs 10 --finetune-epochs 5

# Run all phases including ablations
python run_pipeline.py --phases all
```

### Key Arguments

**Configuration:**
- `--config`: Path to JSON configuration file (overrides command-line arguments)

**Pipeline Phases:**
- `--phases`: Which phases to run (choices: `data`, `train`, `evaluate`, `ablations`, `all`)

**Data:**
- `--cifar10-dir`: CIFAR-10 directory (default: `./data`)
- `--cifar10h-dir`: CIFAR-10H directory (default: `./cifar-10h-1.0.0/data`)

**Output:**
- `--output-dir`: Base output directory (default: `./outputs`)
- `--checkpoint-dir`: Checkpoint directory (default: `./checkpoints`)

**Training:**
- `--pretrain-epochs`: Pretraining epochs (default: 100)
- `--finetune-epochs`: Fine-tuning epochs (default: 50)
- `--loss-functions`: Loss functions to train (default: `kl js custom`)

**Evaluation:**
- `--generate-visualizations`: Generate visualizations (default: True)
- `--evaluate-robustness`: Evaluate robustness to corruptions

**Report:**
- `--generate-report`: Generate comprehensive markdown report (default: True)

**Other:**
- `--device`: Device to use (`cuda` or `cpu`)
- `--seed`: Random seed (default: 42)

### Configuration File

You can use a JSON configuration file to specify all parameters:

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
  "seed": 42,
  "log_level": "INFO"
}
```

Use it with: `python run_pipeline.py --config example_config.json`

### Outputs

- `pipeline_config.json` - Pipeline configuration
- `pipeline_report.md` - Comprehensive markdown report
- All outputs from individual phases (data, training, evaluation, ablations)

## Common Workflows

### Workflow 1: Quick Start

```bash
# 1. Prepare data
python prepare_data.py

# 2. Train models (reduced epochs for testing)
python train.py --pretrain-epochs 10 --finetune-epochs 5

# 3. Evaluate models
python evaluate.py --generate-visualizations
```

### Workflow 2: Full Pipeline

```bash
# Run complete pipeline with all phases
python run_pipeline.py --phases all
```

### Workflow 3: Experiment with Different Loss Functions

```bash
# Train only KL model
python train.py --loss-functions kl

# Evaluate KL model
python evaluate.py --model-path checkpoints/finetuned_kl_best.pth --generate-visualizations
```

### Workflow 4: Ablation Studies

```bash
# Run specific ablation studies
python run_ablations.py --studies loss initialization --pretrain-epochs 20 --finetune-epochs 10
```

### Workflow 5: Custom Configuration

```bash
# Create custom config file
cat > my_config.json << EOF
{
  "pretrain_epochs": 50,
  "finetune_epochs": 25,
  "loss_functions": ["kl"],
  "device": "cpu"
}
EOF

# Run pipeline with custom config
python run_pipeline.py --config my_config.json
```

## Logging

All scripts support configurable logging levels:

```bash
# Debug mode (verbose)
python train.py --log-level DEBUG

# Info mode (default)
python train.py --log-level INFO

# Warning mode (minimal output)
python train.py --log-level WARNING
```

## Error Handling

All scripts include comprehensive error handling:

- **Missing datasets**: Scripts will attempt to download CIFAR-10 automatically
- **Invalid configurations**: Clear error messages with suggestions
- **Training failures**: Checkpoints saved at each epoch, can resume from last checkpoint
- **Evaluation failures**: Individual model failures don't stop batch evaluation

## Performance Tips

1. **Use GPU**: Add `--device cuda` for 10-20x speedup
2. **Reduce epochs for testing**: Use `--pretrain-epochs 10 --finetune-epochs 5` for quick tests
3. **Batch size**: Increase `--batch-size` if you have more GPU memory
4. **Skip pretraining**: Use `--skip-pretrain` if you already have a pretrained model
5. **Parallel evaluation**: Evaluate multiple models simultaneously by running multiple instances

## Troubleshooting

### Issue: CIFAR-10H not found

```bash
# Download CIFAR-10H from https://github.com/jcpeterson/cifar-10h
# Extract to ./cifar-10h-1.0.0/data/
```

### Issue: Out of memory

```bash
# Reduce batch size
python train.py --pretrain-batch-size 64 --finetune-batch-size 32

# Or use CPU
python train.py --device cpu
```

### Issue: Training too slow

```bash
# Use GPU
python train.py --device cuda

# Reduce epochs for testing
python train.py --pretrain-epochs 10 --finetune-epochs 5
```

## Output Directory Structure

After running the complete pipeline:

```
outputs/
├── data_visualizations/
│   ├── entropy_histogram.png
│   ├── per_class_entropy.png
│   ├── example_grid.png
│   ├── data_pipeline_config.json
│   └── split_info.json
├── training_logs/
│   ├── pretrain_history.json
│   ├── finetune_kl_history.json
│   ├── finetune_js_history.json
│   ├── finetune_custom_history.json
│   └── training_config.json
├── evaluation_results/
│   ├── model_comparison.csv
│   ├── finetuned_kl_best/
│   │   ├── evaluation_metrics.json
│   │   ├── per_class_performance.csv
│   │   ├── gradcam_comparison.png
│   │   ├── failure_cases.png
│   │   └── entropy_correlation.png
│   ├── finetuned_js_best/
│   └── finetuned_custom_best/
├── ablation_studies/
│   ├── loss_functions/
│   ├── initialization/
│   ├── training_strategy/
│   └── architecture/
├── pipeline_config.json
└── pipeline_report.md

checkpoints/
├── pretrained_resnet18_cifar10.pth
├── finetuned_kl_best.pth
├── finetuned_js_best.pth
├── finetuned_custom_best.pth
└── ablations/
    └── ...
```

## Additional Resources

- **Main README**: See `README.md` for project overview
- **Design Document**: See `.kiro/specs/cifar10-disagreement-predictor/design.md`
- **Requirements**: See `.kiro/specs/cifar10-disagreement-predictor/requirements.md`
- **Tasks**: See `.kiro/specs/cifar10-disagreement-predictor/tasks.md`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the error messages (they include helpful suggestions)
3. Check the log files in `outputs/training_logs/`
4. Ensure all dependencies are installed: `pip install -r requirements.txt`
