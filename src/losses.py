"""
Loss Functions Module

Implements KL divergence, Jensen-Shannon divergence, and custom entropy-regularized
loss functions for training the disagreement predictor.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def kl_divergence_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Compute KL divergence loss: KL(target || pred).
    
    Args:
        pred_probs: Predicted distributions, shape (batch_size, num_classes)
        target_probs: Target distributions, shape (batch_size, num_classes)
        epsilon: Small constant for numerical stability
    
    Returns:
        loss: Scalar tensor with mean KL divergence across batch
    """
    logger.debug("Computing KL divergence loss")
    
    # Add epsilon to prevent log(0) and division by zero
    pred_probs = pred_probs + epsilon
    target_probs = target_probs + epsilon
    
    # Normalize to ensure valid probability distributions
    pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
    
    # Compute KL divergence: KL(p || q) = Σ p(y) * log(p(y) / q(y))
    kl = target_probs * torch.log(target_probs / pred_probs)
    kl = kl.sum(dim=1)  # Sum over classes
    
    return kl.mean()  # Mean over batch


def js_divergence_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence loss.
    
    JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = 0.5 * (p + q)
    
    Args:
        pred_probs: Predicted distributions, shape (batch_size, num_classes)
        target_probs: Target distributions, shape (batch_size, num_classes)
        epsilon: Small constant for numerical stability
    
    Returns:
        loss: Scalar tensor with mean JS divergence across batch
    """
    logger.debug("Computing JS divergence loss")
    
    # Add epsilon and normalize
    pred_probs = pred_probs + epsilon
    target_probs = target_probs + epsilon
    pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
    
    # Compute mixture distribution
    m = 0.5 * (pred_probs + target_probs)
    
    # Compute KL(target || m) and KL(pred || m)
    kl_target_m = target_probs * torch.log(target_probs / m)
    kl_pred_m = pred_probs * torch.log(pred_probs / m)
    
    # Sum over classes
    kl_target_m = kl_target_m.sum(dim=1)
    kl_pred_m = kl_pred_m.sum(dim=1)
    
    # Compute JS divergence
    js = 0.5 * kl_target_m + 0.5 * kl_pred_m
    
    return js.mean()  # Mean over batch


def compute_entropy(probs: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """
    Compute Shannon entropy for probability distributions.
    
    H(p) = -Σ p(y) * log₂(p(y))
    
    Args:
        probs: Probability distributions, shape (batch_size, num_classes)
        epsilon: Small constant for numerical stability
    
    Returns:
        entropy: Tensor of shape (batch_size,) with entropy values in bits
    """
    logger.debug("Computing Shannon entropy")
    
    # Add epsilon and normalize
    probs = probs + epsilon
    probs = probs / probs.sum(dim=1, keepdim=True)
    
    # Compute entropy in bits (log base 2)
    entropy = -(probs * torch.log2(probs)).sum(dim=1)
    
    return entropy


def custom_entropy_regularized_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    lambda_weight: float = 0.1,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Compute custom loss: KL(p || q) + λ * |H(p) - H(q)|.
    
    Combines distribution matching (KL term) with entropy penalty to explicitly
    encourage correct disagreement level prediction.
    
    Args:
        pred_probs: Predicted distributions, shape (batch_size, num_classes)
        target_probs: Target distributions, shape (batch_size, num_classes)
        lambda_weight: Weight for entropy penalty term (default: 0.1)
        epsilon: Small constant for numerical stability
    
    Returns:
        loss: Scalar tensor with mean loss across batch
    """
    logger.debug("Computing custom entropy-regularized loss")
    
    # Compute KL divergence term
    kl_loss = kl_divergence_loss(pred_probs, target_probs, epsilon)
    
    # Compute entropy for both distributions
    target_entropy = compute_entropy(target_probs, epsilon)
    pred_entropy = compute_entropy(pred_probs, epsilon)
    
    # Compute entropy penalty
    entropy_penalty = torch.abs(target_entropy - pred_entropy).mean()
    
    # Combine terms
    total_loss = kl_loss + lambda_weight * entropy_penalty
    
    return total_loss
