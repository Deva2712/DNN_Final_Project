# CIFAR-10 Human Disagreement Predictor
## Deep Neural Networks — Final Project Report

---

### Team Members

| Name | Roll Number |
|------|-------------|
| Sarthak Amilkanthwar | SE23UCSE019 |
| Sri Hemanshu Dulam | SE23UCSE057 |
| Yashwant Alli | SE23UCSE191 |
| Shahzeeb Mohammad | SE23UCSE197 |
| Swayam Reddy | SE23UCSE169 |

---

## Abstract

This project implements a deep learning system that predicts human annotator disagreement on CIFAR-10 images. Rather than predicting a single hard class label, the system predicts the full probability distribution over labels that reflects how approximately 50 human annotators disagree about image classification. We use the CIFAR-10H dataset and a modified ResNet-18 architecture trained in two stages — pretraining on hard labels followed by fine-tuning on soft labels — with three different divergence-based loss functions.

---

## 1. Introduction

Traditional image classifiers are trained to predict a single "correct" label for each image. However, many real-world images are genuinely ambiguous — a small blurry animal could reasonably be a cat or a dog. Human annotators naturally disagree on such images, and this disagreement contains valuable information about image ambiguity and model uncertainty.

The CIFAR-10H dataset (Peterson et al., ICCV 2019) records how ~50 human annotators label each of the 10,000 CIFAR-10 test images, providing soft probability distributions that capture this natural disagreement.

**Key research questions addressed:**
1. Can a neural network learn to predict the distribution of human opinions rather than a single classification?
2. Which loss function — KL divergence, Jensen-Shannon divergence, or a custom entropy-regularized loss — best captures human disagreement?
3. Does two-stage training (pretrain on hard labels, fine-tune on soft labels) outperform single-stage training?
4. What visual features drive human disagreement? (Grad-CAM analysis)

---

## 2. Dataset

### 2.1 CIFAR-10
- 50,000 training images + 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Image size: 32×32 RGB

### 2.2 CIFAR-10H
- 10,000 images (aligned with CIFAR-10 test set)
- Each image annotated by ~50 human annotators
- Provides raw annotator counts and normalized probability distributions
- Source: https://github.com/jcpeterson/cifar-10h

### 2.3 Dataset Splits (CIFAR-10H)

| Split | Size | Purpose |
|-------|------|---------|
| Train | 6,000 | Fine-tuning on soft labels |
| Validation | 2,000 | Hyperparameter selection, early stopping |
| Test | 2,000 | Final evaluation |

All splits use random seed 42 for reproducibility.

### 2.4 Shannon Entropy Analysis

Shannon entropy H(p) = −Σ p(y) log₂ p(y) quantifies annotator disagreement per image:
- **Low entropy (≈ 0 bits):** All annotators agree on one class
- **High entropy (≈ 3.32 bits):** Annotators are maximally spread across all 10 classes

Most CIFAR-10H images have low entropy (clear images), with a long tail of high-entropy ambiguous images.

---

## 3. Model Architecture

### 3.1 Modified ResNet-18 Backbone

Standard ResNet-18 is designed for 224×224 ImageNet images. For 32×32 CIFAR-10 images, two modifications are applied:

| Modification | Standard ResNet-18 | Modified ResNet-18 |
|---|---|---|
| Initial convolution | 7×7, stride 2 | 3×3, stride 1 |
| Initial pooling | MaxPool 3×3, stride 2 | Removed (Identity) |

This preserves spatial resolution through the early layers, which is critical for small 32×32 inputs.

**Feature dimensions:**

| Layer | Output Shape |
|-------|-------------|
| Input | (B, 3, 32, 32) |
| conv1 (3×3, s=1) | (B, 64, 32, 32) |
| layer1 | (B, 64, 32, 32) |
| layer2 | (B, 128, 16, 16) |
| layer3 | (B, 256, 8, 8) |
| layer4 | (B, 512, 4, 4) |
| avgpool | (B, 512) |

### 3.2 MLP Prediction Head

```
512 → Linear(512, 256) → ReLU → Linear(256, 10) → Softmax
```

The Softmax output is a valid probability distribution over 10 classes, representing predicted annotator disagreement.

**Total parameters:** ~11.13M (backbone ~11M + head ~134K)

---

## 4. Loss Functions

Three loss functions were implemented and compared for the fine-tuning stage:

### 4.1 KL Divergence Loss
```
L_KL = KL(p ‖ q) = Σ p(y) · log(p(y) / q(y))
```
- p: true annotator distribution, q: predicted distribution
- Asymmetric, unbounded
- ε = 1e-7 added for numerical stability

### 4.2 Jensen-Shannon Divergence Loss
```
m = 0.5 · (p + q)
L_JS = 0.5 · KL(p ‖ m) + 0.5 · KL(q ‖ m)
```
- Symmetric version of KL divergence
- Bounded in [0, log 2]
- Smoother gradients than KL

### 4.3 Custom Entropy-Regularized Loss
```
L_custom = KL(p ‖ q) + λ · |H(p) − H(q)|,   λ = 0.1
```
- Combines distribution matching (KL term) with entropy matching
- Explicitly penalizes incorrect disagreement level prediction
- λ = 0.1 balances the two terms

---

## 5. Training Protocol

### 5.1 Two-Stage Training

**Stage 1 — Pretraining on Hard Labels:**

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 train (50,000 images) |
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 100 |
| LR schedule | Cosine annealing |
| Augmentation | RandomHorizontalFlip + RandomCrop(32, padding=4) |

**Stage 2 — Fine-tuning on Soft Labels:**

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10H train (6,000 images) |
| Loss | KL / JS / Custom (3 separate models) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | Patience = 10 (val KL divergence) |
| Augmentation | RandomHorizontalFlip + RandomCrop(32, padding=4) |

### 5.2 Reproducibility
- Random seed 42 for Python, NumPy, PyTorch, and CUDA
- `torch.backends.cudnn.deterministic = True`
- Fixed dataset splits

---

## 6. Evaluation Metrics

### 6.1 Distribution Matching
- **Mean KL Divergence** — measures distribution mismatch (lower is better)
- **Mean JS Divergence** — symmetric bounded divergence (lower is better)
- **Mean Cosine Similarity** — vector similarity between distributions (higher is better)

### 6.2 Entropy Prediction Quality
- **Pearson Correlation** — linear correlation between true and predicted entropy
- **Spearman Correlation** — rank correlation between true and predicted entropy

### 6.3 Ambiguity Ranking
- **Precision@K** (K = 100, 200, 500) — overlap between top-K truly ambiguous images and top-K predicted ambiguous images

### 6.4 Success Criteria

| Metric | Target |
|--------|--------|
| Mean KL Divergence | < 0.5 |
| Pearson r (entropy) | > 0.7 |
| Precision@100 | > 0.6 |

---

## 7. Ablation Studies

Four ablation studies were conducted to validate design choices:

### 7.1 Loss Function Comparison
Compared KL, JS, and Custom entropy-regularized loss across all metrics. Identifies the best training objective for disagreement prediction.

### 7.2 Backbone Initialization
Compared random initialization vs. CIFAR-10 pretraining. Validates the benefit of the two-stage training approach.

### 7.3 Training Strategy
Compared two-stage training (pretrain + finetune) vs. single-stage training (finetune only on soft labels from scratch). Quantifies the contribution of pretraining.

### 7.4 Prediction Head Architecture
Compared single linear layer (512→10) vs. two-layer MLP (512→256→10). Validates the MLP design choice.

---

## 8. Robustness and Explainability

### 8.1 Out-of-Distribution Robustness
Models were evaluated under three types of image corruption at severity levels 1, 3, and 5:
- **Gaussian noise** — random pixel noise
- **Gaussian blur** — spatial blurring
- **Contrast reduction** — reduced image contrast

Entropy change vs. corruption severity was measured to assess prediction stability.

### 8.2 Grad-CAM Visualization
Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to the final convolutional layer to visualize which image regions drive disagreement predictions. Low-entropy (high agreement) and high-entropy (high disagreement) images were compared to understand attention patterns.

### 8.3 Failure Case Analysis
Images with the highest KL divergence between true and predicted distributions were identified and analyzed. For each failure case, the original image, true annotator distribution, predicted distribution, and entropy values were displayed.

### 8.4 Manual Disagreement Categorization
The 20–30 highest-entropy images were manually inspected and categorized by disagreement source:
- Ambiguous identity
- Poor image quality
- Multi-object scene
- Boundary case
- Other

---

## 9. Testing Infrastructure

### 9.1 Property-Based Tests (Hypothesis)

15 correctness properties were implemented using the Hypothesis library, covering all core invariants of the system:

| # | Property | Requirement |
|---|----------|-------------|
| 1 | Probability Distribution Normalization | 2.1, 2.2 |
| 2 | Invalid Distribution Detection | 2.3 |
| 3 | Index-Based Alignment Preservation | 2.4 |
| 4 | Dataset Split Reproducibility | 3.2 |
| 5 | Dataset Split Disjointness | 3.3 |
| 6 | Paired Data Preservation During Splitting | 3.4 |
| 7 | Shannon Entropy Correctness | 4.1 |
| 8 | Entropy Numerical Stability | 4.2 |
| 9 | Entropy Bounds | 4.3 |
| 10 | Data Pipeline Config Round-Trip | 32.3 |
| 11 | Data Pipeline Config Error Reporting | 32.4 |
| 12 | Model Config Round-Trip | 33.3 |
| 13 | Model Config Error Reporting | 33.4 |
| 14 | Training Config Round-Trip | 34.3 |
| 15 | Training Config Error Reporting | 34.4 |

**Result: All 15 property tests pass ✅**

### 9.2 Unit Tests
160 unit tests covering data pipeline, model architecture, loss functions, training utilities, evaluation metrics, and output management.

**Result: 160 tests pass ✅**

### 9.3 Coverage

| Module | Coverage |
|--------|----------|
| `src/losses.py` | 100% |
| `src/model.py` | 98% |
| `src/output_manager.py` | 100% |
| `src/evaluation.py` | 86% |
| `src/data_pipeline.py` | 79% |
| `src/training.py` | 42%* |
| `src/visualization.py` | 7%* |

*Training loops and visualization rendering require actual GPU runs and cannot be fully covered by unit tests.

---

## 10. Project Structure

```
.
├── src/
│   ├── data_pipeline.py      # Data loading, alignment, splitting, entropy
│   ├── model.py              # Modified ResNet-18 + MLP head
│   ├── losses.py             # KL, JS, and custom loss functions
│   ├── training.py           # Two-stage training protocol
│   ├── evaluation.py         # Metrics and ablation comparisons
│   ├── visualization.py      # Plots and Grad-CAM
│   ├── output_manager.py     # Output directory management
│   └── logging_config.py     # Logging setup
│
├── tests/
│   ├── conftest.py           # Shared pytest fixtures
│   ├── property_tests/       # 15 Hypothesis property-based tests
│   ├── test_data_pipeline.py
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_output_manager.py
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── prepare_data.py           # Data preparation script
├── run_ablations.py          # Ablation studies script
├── run_pipeline.py           # End-to-end pipeline script
├── requirements.txt
└── README.md
```

---

## 11. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Prepare data
```bash
python prepare_data.py
```

### Train all three models
```bash
python train.py
```

### Evaluate
```bash
python evaluate.py
```

### Run ablation studies
```bash
python run_ablations.py
```

### Run full pipeline
```bash
python run_pipeline.py
```

### Run tests
```bash
pytest tests/ -v
pytest tests/property_tests/ -v   # Property-based tests only
```

---

## 12. References

1. Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). **Human uncertainty makes classification more robust.** *ICCV 2019*, 9617–9626.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep residual learning for image recognition.** *CVPR 2016*.

3. Krizhevsky, A., Nair, V., & Hinton, G. (2009). **CIFAR-10 (Canadian Institute for Advanced Research).**

4. Loshchilov, I., & Hutter, F. (2019). **Decoupled weight decay regularization.** *ICLR 2019*.

5. Selvaraju, R. R., et al. (2017). **Grad-CAM: Visual explanations from deep networks via gradient-based localization.** *ICCV 2017*.

6. MacKay, D. J. C. (2003). **Information Theory, Inference, and Learning Algorithms.** Cambridge University Press.

---

*Report prepared for Deep Neural Networks course — May 2026*
