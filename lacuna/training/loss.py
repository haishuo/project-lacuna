"""
lacuna.training.loss

Multi-task loss functions for Lacuna training.

Loss Components:
    1. Mechanism Classification Loss:
       - Cross-entropy on mechanism class (MCAR/MAR/MNAR)
       - Cross-entropy on full mechanism (including MNAR variants)
       - Optional: proper scoring rule (Brier score) for calibration
    
    2. Reconstruction Loss:
       - MSE on artificially-masked values
       - Per-head losses for multi-task learning
       - Optional: head-weighted reconstruction loss
    
    3. Auxiliary Losses:
       - MoE load balancing (prevent expert collapse)
       - Entropy regularization (encourage/discourage confidence)
       - KL divergence for latent reconstruction head
    
    4. Combined Loss:
       - Weighted sum of all components
       - Configurable weights for each term
       - Automatic loss scaling (optional)

Training Modes:
    - Pretraining: reconstruction_weight=1.0, mechanism_weight=0.0
    - Classification: reconstruction_weight=0.0, mechanism_weight=1.0
    - Joint: reconstruction_weight>0, mechanism_weight>0

Design Decisions:
    1. All losses return both scalar total and per-component dict
    2. Proper handling of missing labels (reconstruction_mask, class_ids)
    3. Gradient-friendly implementations (no in-place operations)
    4. Support for both class-level and mechanism-level supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field

from lacuna.core.types import (
    TokenBatch,
    PosteriorResult,
    LacunaOutput,
    ReconstructionResult,
    MoEOutput,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LossConfig:
    """Configuration for multi-task loss computation."""
    
    # === Loss Weights ===
    mechanism_weight: float = 1.0       # Weight for mechanism classification loss
    reconstruction_weight: float = 0.5  # Weight for reconstruction loss
    class_weight: float = 0.5           # Weight for class-level (vs mechanism-level) loss
    
    # === Mechanism Loss ===
    mechanism_loss_type: str = "cross_entropy"  # "cross_entropy" or "brier"
    label_smoothing: float = 0.0        # Label smoothing for cross-entropy
    
    # === Reconstruction Loss ===
    reconstruction_loss_type: str = "mse"  # "mse" or "huber"
    huber_delta: float = 1.0            # Delta for Huber loss
    per_head_weights: Optional[Dict[str, float]] = None  # Weights per reconstruction head
    
    # === Auxiliary Losses ===
    load_balance_weight: float = 0.01   # MoE load balancing
    entropy_weight: float = 0.0         # Entropy regularization (positive = encourage entropy)
    kl_weight: float = 0.001            # KL divergence for latent head
    
    # === Loss Scaling ===
    use_loss_scaling: bool = False      # Automatic loss component scaling
    scaling_momentum: float = 0.9       # Momentum for running loss averages
    
    def __post_init__(self):
        if self.mechanism_loss_type not in ("cross_entropy", "brier"):
            raise ValueError(f"Unknown mechanism_loss_type: {self.mechanism_loss_type}")
        
        if self.reconstruction_loss_type not in ("mse", "huber"):
            raise ValueError(f"Unknown reconstruction_loss_type: {self.reconstruction_loss_type}")


# =============================================================================
# Mechanism Classification Losses
# =============================================================================

def mechanism_cross_entropy(
    logits: torch.Tensor,              # [B, n_classes] or [B, n_mechanisms]
    targets: torch.Tensor,             # [B] integer class labels
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy loss for mechanism classification.
    
    Args:
        logits: Raw logits (before softmax). Shape: [B, K]
        targets: Integer class labels. Shape: [B]
        label_smoothing: Label smoothing factor (0 = no smoothing).
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: Scalar loss (if reduction != "none") or [B] per-sample loss.
    """
    return F.cross_entropy(
        logits,
        targets,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def mechanism_cross_entropy_from_probs(
    probs: torch.Tensor,               # [B, n_classes]
    targets: torch.Tensor,             # [B] integer class labels
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy loss computed from probabilities (not logits).
    
    Useful when we have posteriors but not logits (e.g., from MoE).
    Uses log of probabilities as pseudo-logits.
    
    Args:
        probs: Probabilities (after softmax). Shape: [B, K]
        targets: Integer class labels. Shape: [B]
        label_smoothing: Label smoothing factor.
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: Scalar or per-sample loss.
    """
    # Convert probabilities to log-probabilities
    log_probs = torch.log(probs.clamp(min=1e-8))
    
    # NLL loss on log-probabilities
    if label_smoothing > 0:
        # Manual label smoothing for NLL
        n_classes = probs.shape[-1]
        smooth_targets = torch.zeros_like(probs)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_targets = (1 - label_smoothing) * smooth_targets + label_smoothing / n_classes
        loss = -(smooth_targets * log_probs).sum(dim=-1)
    else:
        loss = F.nll_loss(log_probs, targets, reduction="none")
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def brier_score(
    probs: torch.Tensor,               # [B, n_classes]
    targets: torch.Tensor,             # [B] integer class labels
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Brier score for probabilistic predictions.
    
    Brier score is a proper scoring rule that measures calibration:
        BS = (1/K) * sum_k (p_k - y_k)^2
    
    where y_k is 1 for the true class and 0 otherwise.
    
    Args:
        probs: Probabilities. Shape: [B, K]
        targets: Integer class labels. Shape: [B]
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: Brier score (lower is better).
    """
    B, K = probs.shape
    
    # Create one-hot targets
    targets_onehot = F.one_hot(targets, num_classes=K).float()
    
    # Squared error between predicted probs and one-hot targets
    squared_error = (probs - targets_onehot) ** 2
    
    # Average over classes
    brier = squared_error.mean(dim=-1)  # [B]
    
    if reduction == "mean":
        return brier.mean()
    elif reduction == "sum":
        return brier.sum()
    return brier


def class_cross_entropy(
    p_class: torch.Tensor,             # [B, 3] class posterior
    class_targets: torch.Tensor,       # [B] class labels (0=MCAR, 1=MAR, 2=MNAR)
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy on mechanism class (MCAR/MAR/MNAR).
    
    This is the primary classification loss for mechanism identification.
    Uses the aggregated class posterior (MNAR variants summed).
    
    Args:
        p_class: Class posterior. Shape: [B, 3]
        class_targets: Class labels. Shape: [B]
        label_smoothing: Label smoothing factor.
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: Cross-entropy loss.
    """
    return mechanism_cross_entropy_from_probs(
        p_class,
        class_targets,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def mechanism_full_cross_entropy(
    p_mechanism: torch.Tensor,         # [B, n_mechanisms] full mechanism posterior
    mechanism_targets: torch.Tensor,   # [B] mechanism labels (0=MCAR, 1=MAR, 2+=MNAR variants)
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy on full mechanism (including MNAR variants).
    
    Use this when you have variant-level labels from synthetic data.
    
    Args:
        p_mechanism: Full mechanism posterior. Shape: [B, n_mechanisms]
        mechanism_targets: Mechanism labels. Shape: [B]
        label_smoothing: Label smoothing factor.
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: Cross-entropy loss.
    """
    return mechanism_cross_entropy_from_probs(
        p_mechanism,
        mechanism_targets,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


# =============================================================================
# Reconstruction Losses
# =============================================================================

def reconstruction_mse(
    predictions: torch.Tensor,         # [B, max_rows, max_cols]
    targets: torch.Tensor,             # [B, max_rows, max_cols]
    mask: torch.Tensor,                # [B, max_rows, max_cols] cells to evaluate
    row_mask: torch.Tensor,            # [B, max_rows]
    col_mask: torch.Tensor,            # [B, max_cols]
    reduction: str = "mean",
) -> torch.Tensor:
    """
    MSE loss on reconstructed values.
    
    Only computes loss on cells that are:
    1. In the reconstruction mask (artificially masked)
    2. In a valid row
    3. In a valid column
    
    Args:
        predictions: Predicted values. Shape: [B, max_rows, max_cols]
        targets: Ground truth values. Shape: [B, max_rows, max_cols]
        mask: Reconstruction mask. Shape: [B, max_rows, max_cols]
        row_mask: Valid row mask. Shape: [B, max_rows]
        col_mask: Valid column mask. Shape: [B, max_cols]
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: MSE loss.
    """
    # Create full validity mask
    validity = mask & row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)
    
    # Squared error
    squared_error = (predictions - targets) ** 2
    
    # Masked aggregation
    masked_error = squared_error * validity.float()
    
    if reduction == "none":
        return masked_error
    
    total_error = masked_error.sum()
    count = validity.float().sum().clamp(min=1.0)
    
    if reduction == "mean":
        return total_error / count
    return total_error  # sum


def reconstruction_huber(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Huber loss on reconstructed values.
    
    More robust to outliers than MSE.
    
    Args:
        predictions: Predicted values. Shape: [B, max_rows, max_cols]
        targets: Ground truth values. Shape: [B, max_rows, max_cols]
        mask: Reconstruction mask. Shape: [B, max_rows, max_cols]
        row_mask: Valid row mask. Shape: [B, max_rows]
        col_mask: Valid column mask. Shape: [B, max_cols]
        delta: Threshold for switching from L2 to L1.
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: Huber loss.
    """
    # Create full validity mask
    validity = mask & row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)
    
    # Compute Huber loss element-wise
    diff = predictions - targets
    abs_diff = diff.abs()
    
    # Huber: 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
    huber = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta),
    )
    
    # Masked aggregation
    masked_huber = huber * validity.float()
    
    if reduction == "none":
        return masked_huber
    
    total_loss = masked_huber.sum()
    count = validity.float().sum().clamp(min=1.0)
    
    if reduction == "mean":
        return total_loss / count
    return total_loss


def multi_head_reconstruction_loss(
    reconstruction_results: Dict[str, ReconstructionResult],
    original_values: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    head_weights: Optional[Dict[str, float]] = None,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute reconstruction loss across all heads.
    
    Args:
        reconstruction_results: Dict mapping head_name -> ReconstructionResult.
        original_values: Ground truth values. Shape: [B, max_rows, max_cols]
        reconstruction_mask: Cells to evaluate. Shape: [B, max_rows, max_cols]
        row_mask: Valid row mask. Shape: [B, max_rows]
        col_mask: Valid column mask. Shape: [B, max_cols]
        head_weights: Optional per-head weights. Default: uniform.
        loss_type: "mse" or "huber".
        huber_delta: Delta for Huber loss.
    
    Returns:
        total_loss: Weighted sum of per-head losses.
        per_head_losses: Dict mapping head_name -> loss.
    """
    per_head_losses = {}
    
    loss_fn = reconstruction_mse if loss_type == "mse" else reconstruction_huber
    loss_kwargs = {"delta": huber_delta} if loss_type == "huber" else {}
    
    for head_name, result in reconstruction_results.items():
        head_loss = loss_fn(
            predictions=result.predictions,
            targets=original_values,
            mask=reconstruction_mask,
            row_mask=row_mask,
            col_mask=col_mask,
            reduction="mean",
            **loss_kwargs,
        )
        per_head_losses[head_name] = head_loss
    
    # Weighted sum
    if head_weights is None:
        # Uniform weights
        total_loss = sum(per_head_losses.values()) / len(per_head_losses)
    else:
        total_loss = sum(
            head_weights.get(name, 1.0) * loss
            for name, loss in per_head_losses.items()
        )
        total_weight = sum(head_weights.get(name, 1.0) for name in per_head_losses)
        total_loss = total_loss / total_weight
    
    return total_loss, per_head_losses


# =============================================================================
# Auxiliary Losses
# =============================================================================

def kl_divergence_loss(
    mean: torch.Tensor,                # [B, ...] latent mean
    logvar: torch.Tensor,              # [B, ...] latent log-variance
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL divergence from standard normal prior.
    
    Used for VAE-style latent regularization in MNARLatentHead.
    
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    
    Args:
        mean: Latent mean. Shape: [B, ...]
        logvar: Latent log-variance. Shape: [B, ...]
        reduction: "mean", "sum", or "none".
    
    Returns:
        loss: KL divergence.
    """
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    
    # Sum over latent dimensions, keep batch dimension
    kl = kl.sum(dim=tuple(range(1, kl.dim())))  # [B]
    
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


def entropy_loss(
    probs: torch.Tensor,               # [B, K]
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Entropy of probability distribution.
    
    Can be used to encourage (positive weight) or discourage (negative weight)
    confident predictions.
    
    Args:
        probs: Probabilities. Shape: [B, K]
        reduction: "mean", "sum", or "none".
    
    Returns:
        entropy: Entropy (higher = more uncertain).
    """
    log_probs = torch.log(probs.clamp(min=1e-8))
    entropy = -(probs * log_probs).sum(dim=-1)  # [B]
    
    if reduction == "mean":
        return entropy.mean()
    elif reduction == "sum":
        return entropy.sum()
    return entropy


def load_balance_loss(
    gate_probs: torch.Tensor,          # [B, n_experts]
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Load balancing loss for MoE.
    
    Encourages uniform usage of experts across the batch.
    From Switch Transformer: n_experts * sum(f_i * p_i)
    where f_i = fraction of batch routed to expert i
          p_i = average probability assigned to expert i
    
    Args:
        gate_probs: Gating probabilities. Shape: [B, n_experts]
        reduction: "mean" or "sum".
    
    Returns:
        loss: Load balance loss (minimize to encourage uniform usage).
    """
    B, n_experts = gate_probs.shape
    
    # Average probability per expert
    avg_prob = gate_probs.mean(dim=0)  # [n_experts]
    
    # Fraction routed to each expert (hard assignment)
    assignments = gate_probs.argmax(dim=-1)  # [B]
    fractions = torch.zeros(n_experts, device=gate_probs.device)
    for i in range(n_experts):
        fractions[i] = (assignments == i).float().mean()
    
    # Load balance loss
    loss = n_experts * (avg_prob * fractions).sum()
    
    return loss


# =============================================================================
# Combined Loss
# =============================================================================

class LacunaLoss(nn.Module):
    """
    Complete multi-task loss for Lacuna training.
    
    Combines:
        1. Mechanism classification loss (cross-entropy or Brier score)
        2. Reconstruction loss (MSE or Huber)
        3. Auxiliary losses (load balance, entropy, KL)
    
    Attributes:
        config: LossConfig with all loss parameters.
    
    Example:
        >>> loss_fn = LacunaLoss(LossConfig())
        >>> output = model(batch)
        >>> total_loss, loss_dict = loss_fn(output, batch)
        >>> total_loss.backward()
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        # Running averages for loss scaling (if enabled)
        if config.use_loss_scaling:
            self.register_buffer("mechanism_loss_avg", torch.tensor(1.0))
            self.register_buffer("reconstruction_loss_avg", torch.tensor(1.0))
    
    def forward(
        self,
        output: LacunaOutput,
        batch: TokenBatch,
        compute_auxiliary: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute complete multi-task loss.
        
        Args:
            output: LacunaOutput from model forward pass.
            batch: TokenBatch with inputs and labels.
            compute_auxiliary: Whether to compute auxiliary losses.
        
        Returns:
            total_loss: Scalar total loss for backprop.
            loss_dict: Dict with all loss components for logging.
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=output.evidence.device)
        
        # === 1. Mechanism Classification Loss ===
        if self.config.mechanism_weight > 0 and batch.class_ids is not None:
            # Class-level loss (MCAR/MAR/MNAR)
            if self.config.class_weight > 0:
                if self.config.mechanism_loss_type == "cross_entropy":
                    class_loss = class_cross_entropy(
                        output.posterior.p_class,
                        batch.class_ids,
                        label_smoothing=self.config.label_smoothing,
                    )
                else:  # brier
                    class_loss = brier_score(
                        output.posterior.p_class,
                        batch.class_ids,
                    )
                loss_dict["class_loss"] = class_loss
                total_loss = total_loss + self.config.mechanism_weight * self.config.class_weight * class_loss
            
            # Mechanism-level loss (including MNAR variants)
            if self.config.class_weight < 1.0 and batch.variant_ids is not None:
                # Build full mechanism targets
                # variant_ids: -1 for MCAR/MAR, >=0 for MNAR variants
                # Mechanism targets: 0=MCAR, 1=MAR, 2+=MNAR variants
                mechanism_targets = batch.class_ids.clone()
                mnar_mask = batch.class_ids == 2  # MNAR samples
                mechanism_targets[mnar_mask] = 2 + batch.variant_ids[mnar_mask]
                
                if self.config.mechanism_loss_type == "cross_entropy":
                    mechanism_loss = mechanism_full_cross_entropy(
                        output.posterior.p_mechanism,
                        mechanism_targets,
                        label_smoothing=self.config.label_smoothing,
                    )
                else:
                    mechanism_loss = brier_score(
                        output.posterior.p_mechanism,
                        mechanism_targets,
                    )
                loss_dict["mechanism_loss"] = mechanism_loss
                mechanism_contribution = self.config.mechanism_weight * (1 - self.config.class_weight) * mechanism_loss
                total_loss = total_loss + mechanism_contribution
        
        # === 2. Reconstruction Loss ===
        if (
            self.config.reconstruction_weight > 0
            and output.reconstruction is not None
            and batch.reconstruction_mask is not None
            and batch.original_values is not None
        ):
            recon_loss, per_head_losses = multi_head_reconstruction_loss(
                reconstruction_results=output.reconstruction,
                original_values=batch.original_values,
                reconstruction_mask=batch.reconstruction_mask,
                row_mask=batch.row_mask,
                col_mask=batch.col_mask,
                head_weights=self.config.per_head_weights,
                loss_type=self.config.reconstruction_loss_type,
                huber_delta=self.config.huber_delta,
            )
            
            loss_dict["reconstruction_loss"] = recon_loss
            for head_name, head_loss in per_head_losses.items():
                loss_dict[f"recon_{head_name}"] = head_loss
            
            total_loss = total_loss + self.config.reconstruction_weight * recon_loss
        
        # === 3. Auxiliary Losses ===
        if compute_auxiliary:
            # Load balance loss
            if self.config.load_balance_weight > 0 and output.moe_output is not None:
                lb_loss = load_balance_loss(output.moe_output.gate_probs)
                loss_dict["load_balance_loss"] = lb_loss
                total_loss = total_loss + self.config.load_balance_weight * lb_loss
            
            # Entropy regularization
            if self.config.entropy_weight != 0 and output.posterior is not None:
                ent_loss = entropy_loss(output.posterior.p_class)
                loss_dict["entropy"] = ent_loss
                # Positive weight encourages entropy (more uncertain)
                # Negative weight discourages entropy (more confident)
                total_loss = total_loss - self.config.entropy_weight * ent_loss
        
        # === 4. Loss Scaling (optional) ===
        if self.config.use_loss_scaling and self.training:
            self._update_loss_averages(loss_dict)
        
        loss_dict["total_loss"] = total_loss
        
        return total_loss, loss_dict
    
    def _update_loss_averages(self, loss_dict: Dict[str, torch.Tensor]):
        """Update running averages for loss scaling."""
        momentum = self.config.scaling_momentum
        
        if "class_loss" in loss_dict:
            self.mechanism_loss_avg = (
                momentum * self.mechanism_loss_avg
                + (1 - momentum) * loss_dict["class_loss"].detach()
            )
        
        if "reconstruction_loss" in loss_dict:
            self.reconstruction_loss_avg = (
                momentum * self.reconstruction_loss_avg
                + (1 - momentum) * loss_dict["reconstruction_loss"].detach()
            )
    
    def pretraining_loss(
        self,
        output: LacunaOutput,
        batch: TokenBatch,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reconstruction-only loss for pretraining.
        
        Temporarily sets mechanism_weight=0 for pure self-supervised training.
        
        Args:
            output: LacunaOutput from model forward pass.
            batch: TokenBatch with inputs.
        
        Returns:
            total_loss: Reconstruction loss.
            loss_dict: Dict with loss components.
        """
        # Save original weights
        orig_mechanism_weight = self.config.mechanism_weight
        
        # Temporarily disable mechanism loss
        self.config.mechanism_weight = 0.0
        
        try:
            total_loss, loss_dict = self.forward(output, batch, compute_auxiliary=False)
        finally:
            # Restore weights
            self.config.mechanism_weight = orig_mechanism_weight
        
        return total_loss, loss_dict
    
    def classification_loss(
        self,
        output: LacunaOutput,
        batch: TokenBatch,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute classification-only loss.
        
        Temporarily sets reconstruction_weight=0 for pure classification training.
        
        Args:
            output: LacunaOutput from model forward pass.
            batch: TokenBatch with inputs and labels.
        
        Returns:
            total_loss: Classification loss.
            loss_dict: Dict with loss components.
        """
        # Save original weights
        orig_reconstruction_weight = self.config.reconstruction_weight
        
        # Temporarily disable reconstruction loss
        self.config.reconstruction_weight = 0.0
        
        try:
            total_loss, loss_dict = self.forward(output, batch, compute_auxiliary=True)
        finally:
            # Restore weights
            self.config.reconstruction_weight = orig_reconstruction_weight
        
        return total_loss, loss_dict


# =============================================================================
# Accuracy Metrics
# =============================================================================

def compute_class_accuracy(
    p_class: torch.Tensor,             # [B, 3]
    class_targets: torch.Tensor,       # [B]
) -> torch.Tensor:
    """
    Compute classification accuracy on mechanism class.
    
    Args:
        p_class: Class posterior. Shape: [B, 3]
        class_targets: True class labels. Shape: [B]
    
    Returns:
        accuracy: Fraction of correct predictions (scalar).
    """
    predictions = p_class.argmax(dim=-1)
    correct = (predictions == class_targets).float()
    return correct.mean()


def compute_mechanism_accuracy(
    p_mechanism: torch.Tensor,         # [B, n_mechanisms]
    mechanism_targets: torch.Tensor,   # [B]
) -> torch.Tensor:
    """
    Compute classification accuracy on full mechanism.
    
    Args:
        p_mechanism: Full mechanism posterior. Shape: [B, n_mechanisms]
        mechanism_targets: True mechanism labels. Shape: [B]
    
    Returns:
        accuracy: Fraction of correct predictions (scalar).
    """
    predictions = p_mechanism.argmax(dim=-1)
    correct = (predictions == mechanism_targets).float()
    return correct.mean()


def compute_per_class_accuracy(
    p_class: torch.Tensor,             # [B, 3]
    class_targets: torch.Tensor,       # [B]
) -> Dict[str, torch.Tensor]:
    """
    Compute per-class accuracy.
    
    Args:
        p_class: Class posterior. Shape: [B, 3]
        class_targets: True class labels. Shape: [B]
    
    Returns:
        Dict with "mcar_acc", "mar_acc", "mnar_acc".
    """
    predictions = p_class.argmax(dim=-1)
    
    results = {}
    class_names = ["mcar", "mar", "mnar"]
    
    for class_idx, class_name in enumerate(class_names):
        mask = class_targets == class_idx
        if mask.sum() > 0:
            correct = (predictions[mask] == class_targets[mask]).float()
            results[f"{class_name}_acc"] = correct.mean()
        else:
            results[f"{class_name}_acc"] = torch.tensor(float("nan"))
    
    return results


# =============================================================================
# Factory Functions
# =============================================================================

def create_loss_function(
    mechanism_weight: float = 1.0,
    reconstruction_weight: float = 0.5,
    class_weight: float = 0.5,
    mechanism_loss_type: str = "cross_entropy",
    reconstruction_loss_type: str = "mse",
    label_smoothing: float = 0.0,
    load_balance_weight: float = 0.01,
    entropy_weight: float = 0.0,
    per_head_weights: Optional[Dict[str, float]] = None,
) -> LacunaLoss:
    """
    Factory function to create LacunaLoss.
    
    Args:
        mechanism_weight: Weight for mechanism classification loss.
        reconstruction_weight: Weight for reconstruction loss.
        class_weight: Weight for class-level (vs mechanism-level) loss.
        mechanism_loss_type: "cross_entropy" or "brier".
        reconstruction_loss_type: "mse" or "huber".
        label_smoothing: Label smoothing for cross-entropy.
        load_balance_weight: MoE load balancing weight.
        entropy_weight: Entropy regularization weight.
        per_head_weights: Per-reconstruction-head weights.
    
    Returns:
        Configured LacunaLoss instance.
    """
    config = LossConfig(
        mechanism_weight=mechanism_weight,
        reconstruction_weight=reconstruction_weight,
        class_weight=class_weight,
        mechanism_loss_type=mechanism_loss_type,
        reconstruction_loss_type=reconstruction_loss_type,
        label_smoothing=label_smoothing,
        load_balance_weight=load_balance_weight,
        entropy_weight=entropy_weight,
        per_head_weights=per_head_weights,
    )
    
    return LacunaLoss(config)


def create_pretraining_loss() -> LacunaLoss:
    """
    Create loss function for self-supervised pretraining.
    
    Only reconstruction loss, no mechanism classification.
    
    Returns:
        LacunaLoss configured for pretraining.
    """
    return create_loss_function(
        mechanism_weight=0.0,
        reconstruction_weight=1.0,
        load_balance_weight=0.0,
    )


def create_classification_loss() -> LacunaLoss:
    """
    Create loss function for mechanism classification.
    
    Only classification loss, no reconstruction.
    
    Returns:
        LacunaLoss configured for classification.
    """
    return create_loss_function(
        mechanism_weight=1.0,
        reconstruction_weight=0.0,
        load_balance_weight=0.01,
    )


def create_joint_loss(
    mechanism_weight: float = 1.0,
    reconstruction_weight: float = 0.5,
) -> LacunaLoss:
    """
    Create loss function for joint training.
    
    Both classification and reconstruction losses.
    
    Args:
        mechanism_weight: Weight for mechanism loss.
        reconstruction_weight: Weight for reconstruction loss.
    
    Returns:
        LacunaLoss configured for joint training.
    """
    return create_loss_function(
        mechanism_weight=mechanism_weight,
        reconstruction_weight=reconstruction_weight,
        load_balance_weight=0.01,
    )