"""
Unit tests for loss functions.

Tests KL divergence, Jensen-Shannon divergence, and custom entropy-regularized
loss functions for correctness, numerical stability, and edge cases.
"""

import pytest
import torch
import numpy as np
from src.losses import (
    kl_divergence_loss,
    js_divergence_loss,
    compute_entropy,
    custom_entropy_regularized_loss
)


class TestKLDivergenceLoss:
    """Test KL divergence loss function."""
    
    def test_identical_distributions_zero_loss(self):
        """Test that KL(p || p) ≈ 0 for identical distributions."""
        # Create identical distributions
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
        
        loss = kl_divergence_loss(probs, probs)
        
        # Should be very close to zero
        assert loss.item() < 1e-5, f"Expected KL(p||p) ≈ 0, got {loss.item()}"
    
    def test_numerical_stability_with_zeros(self):
        """Test that loss function handles zero probabilities without NaN/Inf."""
        # Create distributions with zeros
        pred_probs = torch.tensor([[0.0, 0.5, 0.5, 0.0], [0.1, 0.0, 0.9, 0.0]])
        target_probs = torch.tensor([[0.2, 0.3, 0.3, 0.2], [0.0, 0.5, 0.5, 0.0]])
        
        loss = kl_divergence_loss(pred_probs, target_probs)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
    
    def test_non_negative(self):
        """Test that KL divergence is always non-negative."""
        # Create random distributions
        pred_probs = torch.softmax(torch.randn(5, 10), dim=1)
        target_probs = torch.softmax(torch.randn(5, 10), dim=1)
        
        loss = kl_divergence_loss(pred_probs, target_probs)
        
        assert loss.item() >= 0, f"KL divergence should be non-negative, got {loss.item()}"
    
    def test_batch_mean(self):
        """Test that loss returns mean across batch."""
        # Create batch of distributions
        pred_probs = torch.tensor([
            [0.1, 0.9],
            [0.5, 0.5],
            [0.9, 0.1]
        ])
        target_probs = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2]
        ])
        
        loss = kl_divergence_loss(pred_probs, target_probs)
        
        # Should return scalar
        assert loss.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), "Loss should be finite"


class TestJSDivergenceLoss:
    """Test Jensen-Shannon divergence loss function."""
    
    def test_symmetry(self):
        """Test that JS(p || q) = JS(q || p) (symmetry property)."""
        # Create two different distributions
        p = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
        q = torch.tensor([[0.4, 0.3, 0.3], [0.1, 0.4, 0.5]])
        
        js_pq = js_divergence_loss(p, q)
        js_qp = js_divergence_loss(q, p)
        
        # Should be equal (within numerical precision)
        assert torch.allclose(js_pq, js_qp, atol=1e-6), \
            f"JS divergence not symmetric: JS(p||q)={js_pq.item()}, JS(q||p)={js_qp.item()}"
    
    def test_identical_distributions_zero_loss(self):
        """Test that JS(p || p) = 0 for identical distributions."""
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
        
        loss = js_divergence_loss(probs, probs)
        
        # Should be very close to zero
        assert loss.item() < 1e-5, f"Expected JS(p||p) = 0, got {loss.item()}"
    
    def test_numerical_stability_with_zeros(self):
        """Test that JS divergence handles zero probabilities without NaN/Inf."""
        pred_probs = torch.tensor([[0.0, 0.5, 0.5, 0.0], [0.1, 0.0, 0.9, 0.0]])
        target_probs = torch.tensor([[0.2, 0.3, 0.3, 0.2], [0.0, 0.5, 0.5, 0.0]])
        
        loss = js_divergence_loss(pred_probs, target_probs)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
    
    def test_bounded(self):
        """Test that JS divergence is bounded [0, log(2)]."""
        # Create random distributions
        pred_probs = torch.softmax(torch.randn(10, 10), dim=1)
        target_probs = torch.softmax(torch.randn(10, 10), dim=1)
        
        loss = js_divergence_loss(pred_probs, target_probs)
        
        # JS divergence should be in [0, log(2)] ≈ [0, 0.693] for natural log
        assert 0 <= loss.item() <= 0.7, \
            f"JS divergence should be in [0, log(2)], got {loss.item()}"


class TestComputeEntropy:
    """Test Shannon entropy computation."""
    
    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        # Uniform distribution over 10 classes
        uniform = torch.ones(1, 10) / 10
        
        entropy = compute_entropy(uniform)
        
        # Maximum entropy for 10 classes: log₂(10) ≈ 3.32 bits
        expected = torch.tensor([np.log2(10)], dtype=torch.float32)
        assert torch.allclose(entropy, expected, atol=1e-5), \
            f"Expected entropy {expected.item():.4f}, got {entropy.item():.4f}"
    
    def test_deterministic_distribution_zero_entropy(self):
        """Test that deterministic distribution has zero entropy."""
        # One-hot distribution (deterministic)
        deterministic = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        
        entropy = compute_entropy(deterministic)
        
        # Should be very close to zero
        assert entropy.item() < 1e-4, \
            f"Expected entropy ≈ 0 for deterministic distribution, got {entropy.item()}"
    
    def test_entropy_bounds(self):
        """Test that entropy is in valid range [0, log₂(num_classes)]."""
        # Random distributions
        probs = torch.softmax(torch.randn(20, 10), dim=1)
        
        entropies = compute_entropy(probs)
        
        # All entropies should be in [0, log₂(10)]
        max_entropy = np.log2(10)
        assert torch.all(entropies >= 0), "Entropy should be non-negative"
        assert torch.all(entropies <= max_entropy + 0.01), \
            f"Entropy should be <= {max_entropy:.4f}, got max {entropies.max().item():.4f}"
    
    def test_numerical_stability_with_zeros(self):
        """Test that entropy computation handles zeros without NaN/Inf."""
        # Distribution with zeros
        probs = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
        
        entropy = compute_entropy(probs)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(entropy), f"Entropy is not finite: {entropy.item()}"
        assert not torch.isnan(entropy), "Entropy is NaN"
        assert not torch.isinf(entropy), "Entropy is Inf"


class TestCustomEntropyRegularizedLoss:
    """Test custom entropy-regularized loss function."""
    
    def test_greater_than_kl_when_entropies_differ(self):
        """Test that custom loss > KL loss when entropies differ significantly."""
        # Create distributions with different entropy levels
        # High entropy (uniform-like)
        high_entropy = torch.tensor([[0.3, 0.3, 0.4]])
        # Low entropy (peaked)
        low_entropy = torch.tensor([[0.8, 0.1, 0.1]])
        
        kl_loss = kl_divergence_loss(high_entropy, low_entropy)
        custom_loss = custom_entropy_regularized_loss(high_entropy, low_entropy)
        
        # Custom loss should be greater due to entropy penalty
        assert custom_loss.item() > kl_loss.item(), \
            f"Custom loss ({custom_loss.item():.4f}) should be > KL loss ({kl_loss.item():.4f})"
    
    def test_approximately_equal_to_kl_when_entropies_match(self):
        """Test that custom loss ≈ KL loss when entropies are similar."""
        # Create distributions with similar entropy but different shapes
        p = torch.tensor([[0.4, 0.3, 0.3]])
        q = torch.tensor([[0.3, 0.4, 0.3]])
        
        kl_loss = kl_divergence_loss(p, q)
        custom_loss = custom_entropy_regularized_loss(p, q)
        
        # Custom loss should be close to KL loss (entropy penalty is small)
        # The difference should be small relative to KL loss
        diff = abs(custom_loss.item() - kl_loss.item())
        assert diff < 0.1 * kl_loss.item(), \
            f"Custom loss ({custom_loss.item():.4f}) should be close to KL loss ({kl_loss.item():.4f})"
    
    def test_numerical_stability(self):
        """Test that custom loss doesn't produce NaN or Inf."""
        # Create distributions with zeros
        pred_probs = torch.tensor([[0.0, 0.5, 0.5, 0.0]])
        target_probs = torch.tensor([[0.2, 0.3, 0.3, 0.2]])
        
        loss = custom_entropy_regularized_loss(pred_probs, target_probs)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
    
    def test_lambda_weight_effect(self):
        """Test that lambda weight controls entropy penalty contribution."""
        # Create distributions with different entropies
        high_entropy = torch.tensor([[0.3, 0.3, 0.4]])
        low_entropy = torch.tensor([[0.8, 0.1, 0.1]])
        
        # Test with different lambda values
        loss_lambda_0 = custom_entropy_regularized_loss(
            high_entropy, low_entropy, lambda_weight=0.0
        )
        loss_lambda_01 = custom_entropy_regularized_loss(
            high_entropy, low_entropy, lambda_weight=0.1
        )
        loss_lambda_05 = custom_entropy_regularized_loss(
            high_entropy, low_entropy, lambda_weight=0.5
        )
        
        # Higher lambda should result in higher loss
        assert loss_lambda_0.item() < loss_lambda_01.item() < loss_lambda_05.item(), \
            "Higher lambda should result in higher loss"


class TestLossFunctionsIntegration:
    """Integration tests for loss functions."""
    
    def test_all_losses_finite_on_random_data(self):
        """Test that all loss functions produce finite values on random data."""
        # Generate random probability distributions
        pred_probs = torch.softmax(torch.randn(32, 10), dim=1)
        target_probs = torch.softmax(torch.randn(32, 10), dim=1)
        
        kl_loss = kl_divergence_loss(pred_probs, target_probs)
        js_loss = js_divergence_loss(pred_probs, target_probs)
        custom_loss = custom_entropy_regularized_loss(pred_probs, target_probs)
        
        # All should be finite
        assert torch.isfinite(kl_loss), "KL loss is not finite"
        assert torch.isfinite(js_loss), "JS loss is not finite"
        assert torch.isfinite(custom_loss), "Custom loss is not finite"
    
    def test_gradient_flow(self):
        """Test that all loss functions allow gradient flow."""
        # Create distributions that require gradients
        pred_logits = torch.randn(8, 10, requires_grad=True)
        pred_probs = torch.softmax(pred_logits, dim=1)
        target_probs = torch.softmax(torch.randn(8, 10), dim=1)
        
        # Test KL loss
        kl_loss = kl_divergence_loss(pred_probs, target_probs)
        kl_loss.backward(retain_graph=True)
        assert pred_logits.grad is not None, "KL loss doesn't allow gradient flow"
        pred_logits.grad.zero_()
        
        # Test JS loss
        js_loss = js_divergence_loss(pred_probs, target_probs)
        js_loss.backward(retain_graph=True)
        assert pred_logits.grad is not None, "JS loss doesn't allow gradient flow"
        pred_logits.grad.zero_()
        
        # Test custom loss
        custom_loss = custom_entropy_regularized_loss(pred_probs, target_probs)
        custom_loss.backward()
        assert pred_logits.grad is not None, "Custom loss doesn't allow gradient flow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
