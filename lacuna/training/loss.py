"""
lacuna.training.loss

Loss functions for Lacuna training.

Primary loss: Cross-entropy on generator classification.
Auxiliary loss: Cross-entropy on class posterior (optional regularization).
"""

import torch
import torch.nn.functional as F
from typing import Optional

from lacuna.core.types import PosteriorResult


def generator_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy loss on generator logits.
    
    Args:
        logits: [B, K] raw generator logits.
        targets: [B] generator IDs (long tensor).
        reduction: "mean", "sum", or "none".
    
    Returns:
        Loss tensor (scalar if reduction != "none").
    """
    return F.cross_entropy(logits, targets, reduction=reduction)


def class_cross_entropy(
    p_class: torch.Tensor,
    class_targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy on class posterior (auxiliary loss).
    
    Note: Uses log of probabilities, not logits.
    
    Args:
        p_class: [B, 3] class posterior probabilities.
        class_targets: [B] class IDs (0=MCAR, 1=MAR, 2=MNAR).
        reduction: "mean", "sum", or "none".
    
    Returns:
        Loss tensor.
    """
    # Add epsilon for numerical stability
    log_p = torch.log(p_class + 1e-10)
    return F.nll_loss(log_p, class_targets, reduction=reduction)


def combined_loss(
    posterior: PosteriorResult,
    generator_targets: torch.Tensor,
    class_targets: torch.Tensor,
    class_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss with optional class auxiliary term.
    
    Total loss = CE(generator) + class_weight * CE(class)
    
    Args:
        posterior: Model output containing logits and p_class.
        generator_targets: [B] generator IDs.
        class_targets: [B] class IDs.
        class_weight: Weight for auxiliary class loss (0 = disabled).
    
    Returns:
        total_loss: Combined loss tensor.
        metrics: Dict with individual loss components.
    """
    gen_loss = generator_cross_entropy(posterior.logits_generator, generator_targets)
    
    metrics = {"loss_generator": gen_loss.item()}
    
    if class_weight > 0:
        cls_loss = class_cross_entropy(posterior.p_class, class_targets)
        total_loss = gen_loss + class_weight * cls_loss
        metrics["loss_class"] = cls_loss.item()
        metrics["loss_total"] = total_loss.item()
    else:
        total_loss = gen_loss
        metrics["loss_total"] = gen_loss.item()
    
    return total_loss, metrics


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute classification accuracy.
    
    Args:
        logits: [B, K] prediction logits.
        targets: [B] ground truth labels.
    
    Returns:
        Accuracy as float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def compute_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 3,
) -> float:
    """Compute top-k accuracy.
    
    Args:
        logits: [B, K] prediction logits.
        targets: [B] ground truth labels.
        k: Number of top predictions to consider.
    
    Returns:
        Top-k accuracy as float in [0, 1].
    """
    B, K = logits.shape
    k = min(k, K)
    
    _, topk_preds = logits.topk(k, dim=-1)  # [B, k]
    correct = (topk_preds == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()
