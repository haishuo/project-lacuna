"""
lacuna.models.aggregator

Aggregate generator posterior to class posterior.

The key insight: we train on generators but evaluate on classes.
π(c|Z) = Σ_{k: κ(k)=c} p(G_k|Z)

where κ maps generator IDs to class IDs.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def aggregate_to_class_posterior(
    p_generator: torch.Tensor,
    class_mapping: torch.Tensor,
) -> torch.Tensor:
    """Aggregate generator posterior to class posterior.
    
    Args:
        p_generator: [B, K] generator posterior probabilities.
        class_mapping: [K] tensor mapping generator_id -> class_id.
    
    Returns:
        [B, 3] class posterior (MCAR, MAR, MNAR).
    """
    B, K = p_generator.shape
    n_classes = 3
    
    # Initialize class posterior
    p_class = torch.zeros(B, n_classes, device=p_generator.device, dtype=p_generator.dtype)
    
    # Sum probabilities for each class
    for c in range(n_classes):
        # Mask for generators belonging to class c
        mask = (class_mapping == c)  # [K]
        if mask.any():
            p_class[:, c] = p_generator[:, mask].sum(dim=1)
    
    return p_class


def aggregate_to_class_posterior_efficient(
    p_generator: torch.Tensor,
    class_mapping: torch.Tensor,
) -> torch.Tensor:
    """Efficient aggregation using scatter_add.
    
    Same as aggregate_to_class_posterior but uses scatter for speed.
    """
    B, K = p_generator.shape
    n_classes = 3
    device = p_generator.device
    dtype = p_generator.dtype
    
    # Create aggregation matrix [K, 3]
    # agg_matrix[k, c] = 1 if generator k belongs to class c
    agg_matrix = torch.zeros(K, n_classes, device=device, dtype=dtype)
    agg_matrix[torch.arange(K, device=device), class_mapping] = 1.0
    
    # Matrix multiply: [B, K] @ [K, 3] -> [B, 3]
    p_class = p_generator @ agg_matrix
    
    return p_class


def compute_entropy(
    p: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute entropy along specified dimension.
    
    H(p) = -Σ p_i log(p_i)
    
    Args:
        p: Probability tensor.
        dim: Dimension to compute entropy over.
        eps: Small value to avoid log(0).
    
    Returns:
        Entropy tensor with `dim` reduced.
    """
    p_safe = p.clamp(min=eps)
    return -(p_safe * p_safe.log()).sum(dim=dim)


def compute_confidence(
    p_class: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute confidence score from class posterior.
    
    Confidence = 1 - H(p) / H_max
    
    where H_max = log(3) for 3 classes.
    
    Args:
        p_class: [B, 3] class posterior.
    
    Returns:
        [B] confidence scores in [0, 1].
    """
    entropy = compute_entropy(p_class, dim=-1, eps=eps)
    max_entropy = torch.log(torch.tensor(3.0, device=p_class.device))
    confidence = 1.0 - entropy / max_entropy
    return confidence.clamp(0, 1)


def get_predicted_class(p_class: torch.Tensor) -> torch.Tensor:
    """Get predicted class from posterior.
    
    Args:
        p_class: [B, 3] class posterior.
    
    Returns:
        [B] predicted class IDs (0=MCAR, 1=MAR, 2=MNAR).
    """
    return p_class.argmax(dim=-1)
