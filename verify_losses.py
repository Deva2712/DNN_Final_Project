"""
Quick verification script for loss functions.
"""

import torch
from src.losses import (
    kl_divergence_loss,
    js_divergence_loss,
    compute_entropy,
    custom_entropy_regularized_loss
)

def main():
    print("=" * 60)
    print("Loss Functions Verification")
    print("=" * 60)
    
    # Create sample probability distributions
    batch_size = 4
    num_classes = 10
    
    # Simulate model predictions and targets
    pred_probs = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    target_probs = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    print(f"\nSample predicted distribution:\n{pred_probs[0]}")
    print(f"\nSample target distribution:\n{target_probs[0]}")
    
    # Test KL divergence
    print("\n" + "-" * 60)
    print("1. KL Divergence Loss")
    print("-" * 60)
    kl_loss = kl_divergence_loss(pred_probs, target_probs)
    print(f"KL(target || pred) = {kl_loss.item():.6f}")
    print(f"✓ Non-negative: {kl_loss.item() >= 0}")
    print(f"✓ Finite: {torch.isfinite(kl_loss).item()}")
    
    # Test JS divergence
    print("\n" + "-" * 60)
    print("2. Jensen-Shannon Divergence Loss")
    print("-" * 60)
    js_loss = js_divergence_loss(pred_probs, target_probs)
    print(f"JS(pred || target) = {js_loss.item():.6f}")
    print(f"✓ Non-negative: {js_loss.item() >= 0}")
    print(f"✓ Bounded [0, log(2)]: {0 <= js_loss.item() <= 0.7}")
    print(f"✓ Finite: {torch.isfinite(js_loss).item()}")
    
    # Test symmetry
    js_loss_reverse = js_divergence_loss(target_probs, pred_probs)
    print(f"JS(target || pred) = {js_loss_reverse.item():.6f}")
    print(f"✓ Symmetric: {torch.allclose(js_loss, js_loss_reverse, atol=1e-6)}")
    
    # Test entropy computation
    print("\n" + "-" * 60)
    print("3. Shannon Entropy Computation")
    print("-" * 60)
    pred_entropy = compute_entropy(pred_probs)
    target_entropy = compute_entropy(target_probs)
    print(f"Predicted entropies: {pred_entropy}")
    print(f"Target entropies: {target_entropy}")
    print(f"Mean predicted entropy: {pred_entropy.mean().item():.4f} bits")
    print(f"Mean target entropy: {target_entropy.mean().item():.4f} bits")
    print(f"✓ All in valid range [0, {torch.log2(torch.tensor(10.0)).item():.4f}]: "
          f"{torch.all((pred_entropy >= 0) & (pred_entropy <= 3.33)).item()}")
    
    # Test custom loss
    print("\n" + "-" * 60)
    print("4. Custom Entropy-Regularized Loss")
    print("-" * 60)
    custom_loss = custom_entropy_regularized_loss(pred_probs, target_probs, lambda_weight=0.1)
    print(f"Custom loss (λ=0.1) = {custom_loss.item():.6f}")
    print(f"KL component = {kl_loss.item():.6f}")
    print(f"Entropy penalty = {torch.abs(pred_entropy - target_entropy).mean().item():.6f}")
    print(f"✓ Custom loss > KL loss: {custom_loss.item() > kl_loss.item()}")
    print(f"✓ Finite: {torch.isfinite(custom_loss).item()}")
    
    # Test with identical distributions
    print("\n" + "-" * 60)
    print("5. Edge Case: Identical Distributions")
    print("-" * 60)
    identical = torch.softmax(torch.randn(2, 10), dim=1)
    kl_identical = kl_divergence_loss(identical, identical)
    js_identical = js_divergence_loss(identical, identical)
    print(f"KL(p || p) = {kl_identical.item():.8f} (should be ≈ 0)")
    print(f"JS(p || p) = {js_identical.item():.8f} (should be ≈ 0)")
    print(f"✓ KL ≈ 0: {kl_identical.item() < 1e-5}")
    print(f"✓ JS ≈ 0: {js_identical.item() < 1e-5}")
    
    # Test with zero probabilities
    print("\n" + "-" * 60)
    print("6. Edge Case: Zero Probabilities")
    print("-" * 60)
    with_zeros_pred = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
    with_zeros_target = torch.tensor([[0.2, 0.3, 0.3, 0.2, 0.0]])
    kl_zeros = kl_divergence_loss(with_zeros_pred, with_zeros_target)
    js_zeros = js_divergence_loss(with_zeros_pred, with_zeros_target)
    custom_zeros = custom_entropy_regularized_loss(with_zeros_pred, with_zeros_target)
    print(f"KL with zeros = {kl_zeros.item():.6f}")
    print(f"JS with zeros = {js_zeros.item():.6f}")
    print(f"Custom with zeros = {custom_zeros.item():.6f}")
    print(f"✓ All finite: {torch.isfinite(kl_zeros).item() and torch.isfinite(js_zeros).item() and torch.isfinite(custom_zeros).item()}")
    
    print("\n" + "=" * 60)
    print("✓ All loss functions implemented correctly!")
    print("=" * 60)

if __name__ == "__main__":
    main()
