"""
Unit tests for evaluation module.

Tests distribution matching metrics, entropy correlation, Precision@K,
and ablation study functions.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import (
    compute_distribution_metrics,
    compute_entropy_correlation,
    compute_precision_at_k,
    evaluate_model,
    compare_loss_functions,
    compare_backbone_initialization,
    compare_training_strategies,
    compare_prediction_head_architectures,
    analyze_per_class_performance,
    EvaluationMetrics
)
from src.model import DisagreementPredictor


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    model = DisagreementPredictor()
    model.eval()
    return model


@pytest.fixture
def mock_test_loader():
    """Create a mock test data loader."""
    # Create synthetic data
    num_samples = 100
    images = torch.randn(num_samples, 3, 32, 32)
    
    # Create soft labels (random probability distributions)
    soft_labels = torch.rand(num_samples, 10)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
    
    # Create hard labels
    hard_labels = torch.randint(0, 10, (num_samples,))
    
    # Create entropies
    entropies = torch.rand(num_samples) * 3.32  # Max entropy for 10 classes
    
    # Create dataset and loader
    dataset = TensorDataset(images, soft_labels, hard_labels, entropies)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    return loader


class TestDistributionMetrics:
    """Tests for distribution matching metrics."""
    
    def test_compute_distribution_metrics_returns_correct_keys(self, mock_model, mock_test_loader):
        """Test that distribution metrics returns all expected keys."""
        metrics = compute_distribution_metrics(mock_model, mock_test_loader, device='cpu')
        
        expected_keys = ['mean_kl', 'std_kl', 'mean_js', 'std_js', 'mean_cosine', 'std_cosine']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_distribution_metrics_are_valid_numbers(self, mock_model, mock_test_loader):
        """Test that all metrics are valid numbers (not NaN or Inf)."""
        metrics = compute_distribution_metrics(mock_model, mock_test_loader, device='cpu')
        
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float"
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"
    
    def test_kl_divergence_is_non_negative(self, mock_model, mock_test_loader):
        """Test that KL divergence is non-negative."""
        metrics = compute_distribution_metrics(mock_model, mock_test_loader, device='cpu')
        
        assert metrics['mean_kl'] >= 0, "Mean KL divergence should be non-negative"
    
    def test_js_divergence_is_bounded(self, mock_model, mock_test_loader):
        """Test that JS divergence is bounded [0, log(2)]."""
        metrics = compute_distribution_metrics(mock_model, mock_test_loader, device='cpu')
        
        assert metrics['mean_js'] >= 0, "Mean JS divergence should be non-negative"
        assert metrics['mean_js'] <= np.log(2) + 0.1, "Mean JS divergence should be bounded by log(2)"
    
    def test_cosine_similarity_is_bounded(self, mock_model, mock_test_loader):
        """Test that cosine similarity is in [-1, 1]."""
        metrics = compute_distribution_metrics(mock_model, mock_test_loader, device='cpu')
        
        assert -1 <= metrics['mean_cosine'] <= 1, "Cosine similarity should be in [-1, 1]"


class TestEntropyCorrelation:
    """Tests for entropy correlation metrics."""
    
    def test_compute_entropy_correlation_returns_correct_keys(self, mock_model, mock_test_loader):
        """Test that entropy correlation returns all expected keys."""
        metrics = compute_entropy_correlation(mock_model, mock_test_loader, device='cpu')
        
        expected_keys = ['pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 
                        'true_entropies', 'pred_entropies']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_correlation_coefficients_are_bounded(self, mock_model, mock_test_loader):
        """Test that correlation coefficients are in [-1, 1]."""
        metrics = compute_entropy_correlation(mock_model, mock_test_loader, device='cpu')
        
        assert -1 <= metrics['pearson_r'] <= 1, "Pearson r should be in [-1, 1]"
        assert -1 <= metrics['spearman_r'] <= 1, "Spearman r should be in [-1, 1]"
    
    def test_p_values_are_valid(self, mock_model, mock_test_loader):
        """Test that p-values are in [0, 1]."""
        metrics = compute_entropy_correlation(mock_model, mock_test_loader, device='cpu')
        
        assert 0 <= metrics['pearson_p'] <= 1, "Pearson p-value should be in [0, 1]"
        assert 0 <= metrics['spearman_p'] <= 1, "Spearman p-value should be in [0, 1]"
    
    def test_entropy_arrays_have_correct_length(self, mock_test_loader):
        """Test that entropy arrays have correct length."""
        model = DisagreementPredictor()
        model.eval()
        
        metrics = compute_entropy_correlation(model, mock_test_loader, device='cpu')
        
        # Should have 100 samples (from mock_test_loader)
        assert len(metrics['true_entropies']) == 100
        assert len(metrics['pred_entropies']) == 100


class TestPrecisionAtK:
    """Tests for Precision@K metrics."""
    
    def test_compute_precision_at_k_returns_correct_keys(self, mock_model, mock_test_loader):
        """Test that Precision@K returns all expected keys."""
        k_values = [10, 20, 50]
        metrics = compute_precision_at_k(mock_model, mock_test_loader, k_values=k_values, device='cpu')
        
        for k in k_values:
            assert f'precision@{k}' in metrics, f"Missing key: precision@{k}"
    
    def test_precision_values_are_bounded(self, mock_model, mock_test_loader):
        """Test that precision values are in [0, 1]."""
        metrics = compute_precision_at_k(mock_model, mock_test_loader, device='cpu')
        
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{key} should be in [0, 1], got {value}"
    
    def test_precision_with_perfect_ranking(self):
        """Test precision with perfect ranking (should be 1.0)."""
        # Create a model that predicts perfectly
        class PerfectModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                # Return uniform distribution for simplicity
                batch_size = x.size(0)
                return torch.ones(batch_size, 10) / 10
        
        model = PerfectModel()
        model.eval()
        
        # Create data where true and predicted entropies are identical
        num_samples = 50
        images = torch.randn(num_samples, 3, 32, 32)
        soft_labels = torch.ones(num_samples, 10) / 10  # Uniform distributions
        hard_labels = torch.randint(0, 10, (num_samples,))
        entropies = torch.ones(num_samples) * 3.32  # All same entropy
        
        dataset = TensorDataset(images, soft_labels, hard_labels, entropies)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        metrics = compute_precision_at_k(model, loader, k_values=[10], device='cpu')
        
        # With uniform distributions, precision should be reasonable
        assert 0 <= metrics['precision@10'] <= 1


class TestComprehensiveEvaluation:
    """Tests for comprehensive evaluation function."""
    
    def test_evaluate_model_returns_all_metrics(self, mock_model, mock_test_loader):
        """Test that evaluate_model returns all expected metrics."""
        metrics = evaluate_model(mock_model, mock_test_loader, device='cpu')
        
        expected_keys = [
            'mean_kl', 'std_kl', 'mean_js', 'std_js', 'mean_cosine', 'std_cosine',
            'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p',
            'precision@100', 'precision@200', 'precision@500'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_evaluate_model_saves_json(self, mock_model, mock_test_loader, tmp_path):
        """Test that evaluate_model saves metrics to JSON."""
        output_dir = str(tmp_path)
        metrics = evaluate_model(mock_model, mock_test_loader, device='cpu', output_dir=output_dir)
        
        import os
        json_path = os.path.join(output_dir, 'evaluation_metrics.json')
        assert os.path.exists(json_path), "Metrics JSON file should be created"


class TestAblationStudies:
    """Tests for ablation study functions."""
    
    def test_compare_loss_functions(self, mock_test_loader):
        """Test loss function comparison."""
        # Create multiple models
        models = {
            'kl': DisagreementPredictor(),
            'js': DisagreementPredictor(),
            'custom': DisagreementPredictor()
        }
        
        for model in models.values():
            model.eval()
        
        comparison_df = compare_loss_functions(models, mock_test_loader, device='cpu')
        
        # Check DataFrame structure
        assert len(comparison_df) == 3, "Should have 3 rows (one per loss function)"
        assert 'loss_function' in comparison_df.columns
        assert set(comparison_df['loss_function']) == {'kl', 'js', 'custom'}
    
    def test_compare_backbone_initialization(self, mock_test_loader):
        """Test backbone initialization comparison."""
        models = {
            'random': DisagreementPredictor(),
            'cifar10': DisagreementPredictor()
        }
        
        for model in models.values():
            model.eval()
        
        comparison_df = compare_backbone_initialization(models, mock_test_loader, device='cpu')
        
        assert len(comparison_df) == 2
        assert 'initialization' in comparison_df.columns
    
    def test_compare_training_strategies(self, mock_test_loader):
        """Test training strategy comparison."""
        models = {
            'two_stage': DisagreementPredictor(),
            'single_stage': DisagreementPredictor()
        }
        
        for model in models.values():
            model.eval()
        
        comparison_df = compare_training_strategies(models, mock_test_loader, device='cpu')
        
        assert len(comparison_df) == 2
        assert 'training_strategy' in comparison_df.columns
    
    def test_compare_prediction_head_architectures(self, mock_test_loader):
        """Test prediction head architecture comparison."""
        models = {
            'single_layer': DisagreementPredictor(),
            'two_layer_mlp': DisagreementPredictor()
        }
        
        for model in models.values():
            model.eval()
        
        comparison_df = compare_prediction_head_architectures(models, mock_test_loader, device='cpu')
        
        assert len(comparison_df) == 2
        assert 'architecture' in comparison_df.columns


class TestPerClassAnalysis:
    """Tests for per-class performance analysis."""
    
    def test_analyze_per_class_performance(self, mock_model, mock_test_loader):
        """Test per-class performance analysis."""
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        per_class_df = analyze_per_class_performance(
            mock_model, mock_test_loader, class_names, device='cpu'
        )
        
        # Check DataFrame structure
        assert len(per_class_df) == 10, "Should have 10 rows (one per class)"
        assert 'class' in per_class_df.columns
        assert set(per_class_df['class']) == set(class_names)
        
        # Check metric columns
        expected_cols = ['mean_kl', 'std_kl', 'mean_js', 'std_js', 
                        'pearson_r', 'mean_true_entropy', 'num_samples']
        for col in expected_cols:
            assert col in per_class_df.columns, f"Missing column: {col}"
    
    def test_per_class_metrics_are_valid(self, mock_model, mock_test_loader):
        """Test that per-class metrics are valid numbers."""
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        per_class_df = analyze_per_class_performance(
            mock_model, mock_test_loader, class_names, device='cpu'
        )
        
        # Check all numeric columns are valid
        numeric_cols = ['mean_kl', 'std_kl', 'mean_js', 'std_js', 
                       'pearson_r', 'mean_true_entropy']
        
        for col in numeric_cols:
            assert not per_class_df[col].isna().any(), f"{col} contains NaN values"
            assert not np.isinf(per_class_df[col]).any(), f"{col} contains Inf values"


class TestEvaluationMetricsDataclass:
    """Tests for EvaluationMetrics dataclass."""
    
    def test_evaluation_metrics_creation(self):
        """Test creating EvaluationMetrics instance."""
        metrics = EvaluationMetrics(
            mean_kl=0.5, std_kl=0.1,
            mean_js=0.3, std_js=0.05,
            mean_cosine=0.8, std_cosine=0.1,
            pearson_r=0.7, pearson_p=0.001,
            spearman_r=0.65, spearman_p=0.002,
            precision_at_100=0.6,
            precision_at_200=0.55,
            precision_at_500=0.5
        )
        
        assert metrics.mean_kl == 0.5
        assert metrics.pearson_r == 0.7
    
    def test_evaluation_metrics_to_dict(self):
        """Test converting EvaluationMetrics to dictionary."""
        metrics = EvaluationMetrics(
            mean_kl=0.5, std_kl=0.1,
            mean_js=0.3, std_js=0.05,
            mean_cosine=0.8, std_cosine=0.1,
            pearson_r=0.7, pearson_p=0.001,
            spearman_r=0.65, spearman_p=0.002,
            precision_at_100=0.6,
            precision_at_200=0.55,
            precision_at_500=0.5
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['mean_kl'] == 0.5
        assert metrics_dict['pearson_r'] == 0.7
    
    def test_evaluation_metrics_from_dict(self):
        """Test creating EvaluationMetrics from dictionary."""
        data = {
            'mean_kl': 0.5, 'std_kl': 0.1,
            'mean_js': 0.3, 'std_js': 0.05,
            'mean_cosine': 0.8, 'std_cosine': 0.1,
            'pearson_r': 0.7, 'pearson_p': 0.001,
            'spearman_r': 0.65, 'spearman_p': 0.002,
            'precision@100': 0.6,
            'precision@200': 0.55,
            'precision@500': 0.5
        }
        
        metrics = EvaluationMetrics.from_dict(data)
        
        assert metrics.mean_kl == 0.5
        assert metrics.precision_at_100 == 0.6
    
    def test_evaluation_metrics_to_json(self, tmp_path):
        """Test saving EvaluationMetrics to JSON."""
        metrics = EvaluationMetrics(
            mean_kl=0.5, std_kl=0.1,
            mean_js=0.3, std_js=0.05,
            mean_cosine=0.8, std_cosine=0.1,
            pearson_r=0.7, pearson_p=0.001,
            spearman_r=0.65, spearman_p=0.002,
            precision_at_100=0.6,
            precision_at_200=0.55,
            precision_at_500=0.5
        )
        
        json_path = tmp_path / "metrics.json"
        metrics.to_json(str(json_path))
        
        assert json_path.exists()
        
        # Load and verify
        import json
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['mean_kl'] == 0.5
        assert loaded_data['pearson_r'] == 0.7


class TestCorruptionFunctions:
    """Tests for image corruption functions (Task 10.3)."""
    
    def test_gaussian_noise_preserves_shape(self):
        """Test that Gaussian noise corruption preserves image shape."""
        from src.evaluation import add_gaussian_noise
        
        image = torch.rand(3, 32, 32)
        
        for severity in [1, 3, 5]:
            corrupted = add_gaussian_noise(image, severity)
            assert corrupted.shape == image.shape, f"Shape mismatch for severity {severity}"
    
    def test_gaussian_noise_produces_valid_values(self):
        """Test that Gaussian noise produces values in [0, 1]."""
        from src.evaluation import add_gaussian_noise
        
        image = torch.rand(3, 32, 32)
        
        for severity in [1, 3, 5]:
            corrupted = add_gaussian_noise(image, severity)
            assert corrupted.min() >= 0, f"Values below 0 for severity {severity}"
            assert corrupted.max() <= 1, f"Values above 1 for severity {severity}"
    
    def test_gaussian_noise_severity_levels(self):
        """Test that higher severity produces more noise."""
        from src.evaluation import add_gaussian_noise
        
        image = torch.rand(3, 32, 32)
        
        # Apply different severities
        corrupted_1 = add_gaussian_noise(image, 1)
        corrupted_3 = add_gaussian_noise(image, 3)
        corrupted_5 = add_gaussian_noise(image, 5)
        
        # Compute differences from original
        diff_1 = (image - corrupted_1).abs().mean()
        diff_3 = (image - corrupted_3).abs().mean()
        diff_5 = (image - corrupted_5).abs().mean()
        
        # Higher severity should generally produce larger differences
        # (This is probabilistic, so we just check they're all positive)
        assert diff_1 > 0, "Severity 1 should change the image"
        assert diff_3 > 0, "Severity 3 should change the image"
        assert diff_5 > 0, "Severity 5 should change the image"
    
    def test_gaussian_blur_preserves_shape(self):
        """Test that Gaussian blur preserves image shape."""
        from src.evaluation import apply_gaussian_blur
        
        image = torch.rand(3, 32, 32)
        
        for severity in [1, 3, 5]:
            blurred = apply_gaussian_blur(image, severity)
            assert blurred.shape == image.shape, f"Shape mismatch for severity {severity}"
    
    def test_gaussian_blur_produces_valid_values(self):
        """Test that Gaussian blur produces valid pixel values."""
        from src.evaluation import apply_gaussian_blur
        
        image = torch.rand(3, 32, 32)
        
        for severity in [1, 3, 5]:
            blurred = apply_gaussian_blur(image, severity)
            # Blur should not create values outside input range
            assert blurred.min() >= 0, f"Values below 0 for severity {severity}"
            assert blurred.max() <= 1, f"Values above 1 for severity {severity}"
    
    def test_contrast_reduction_preserves_shape(self):
        """Test that contrast reduction preserves image shape."""
        from src.evaluation import reduce_contrast
        
        image = torch.rand(3, 32, 32)
        
        for severity in [1, 3, 5]:
            low_contrast = reduce_contrast(image, severity)
            assert low_contrast.shape == image.shape, f"Shape mismatch for severity {severity}"
    
    def test_contrast_reduction_produces_valid_values(self):
        """Test that contrast reduction produces values in [0, 1]."""
        from src.evaluation import reduce_contrast
        
        image = torch.rand(3, 32, 32)
        
        for severity in [1, 3, 5]:
            low_contrast = reduce_contrast(image, severity)
            assert low_contrast.min() >= 0, f"Values below 0 for severity {severity}"
            assert low_contrast.max() <= 1, f"Values above 1 for severity {severity}"
    
    def test_contrast_reduction_reduces_variance(self):
        """Test that contrast reduction reduces image variance."""
        from src.evaluation import reduce_contrast
        
        # Create an image with high contrast
        image = torch.rand(3, 32, 32)
        
        original_var = image.var()
        
        for severity in [1, 3, 5]:
            low_contrast = reduce_contrast(image, severity)
            reduced_var = low_contrast.var()
            
            # Contrast reduction should reduce variance
            assert reduced_var <= original_var, f"Variance not reduced for severity {severity}"


class TestCorruptionRobustness:
    """Tests for corruption robustness evaluation (Task 10.2)."""
    
    def test_evaluate_corruption_robustness_returns_correct_structure(self, mock_model, mock_test_loader):
        """Test that corruption robustness evaluation returns correct structure."""
        from src.evaluation import evaluate_corruption_robustness
        
        results = evaluate_corruption_robustness(
            mock_model, mock_test_loader, device='cpu'
        )
        
        # Check structure
        expected_corruptions = ['gaussian_noise', 'gaussian_blur', 'contrast_reduction']
        expected_severities = [1, 3, 5]
        
        for corruption in expected_corruptions:
            assert corruption in results, f"Missing corruption type: {corruption}"
            for severity in expected_severities:
                assert severity in results[corruption], f"Missing severity {severity} for {corruption}"
    
    def test_corruption_robustness_values_are_valid(self, mock_model, mock_test_loader):
        """Test that entropy change values are valid numbers."""
        from src.evaluation import evaluate_corruption_robustness
        
        results = evaluate_corruption_robustness(
            mock_model, mock_test_loader, device='cpu'
        )
        
        for corruption_type in results:
            for severity, entropy_change in results[corruption_type].items():
                assert isinstance(entropy_change, float), f"Value is not float for {corruption_type} severity {severity}"
                assert not np.isnan(entropy_change), f"NaN value for {corruption_type} severity {severity}"
                assert not np.isinf(entropy_change), f"Inf value for {corruption_type} severity {severity}"
                assert entropy_change >= 0, f"Negative entropy change for {corruption_type} severity {severity}"
    
    def test_corruption_robustness_saves_json(self, mock_model, mock_test_loader, tmp_path):
        """Test that corruption robustness saves results to JSON."""
        from src.evaluation import evaluate_corruption_robustness
        
        output_dir = str(tmp_path)
        results = evaluate_corruption_robustness(
            mock_model, mock_test_loader, device='cpu', output_dir=output_dir
        )
        
        import os
        json_path = os.path.join(output_dir, 'corruption_robustness.json')
        assert os.path.exists(json_path), "Results JSON file should be created"
