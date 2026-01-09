"""
lacuna.models.decision

Bayes-optimal decision rule under loss matrix.

The decision rule maps class posterior to an action that minimizes
expected loss:

a*(Z) = argmin_a Σ_c L[a,c] π(c|Z)

Actions:
- Green (0): Proceed with MAR assumption
- Yellow (1): Proceed with caution / sensitivity analysis
- Red (2): Do not assume MAR without sensitivity analysis
"""

import torch
from typing import Tuple

from lacuna.core.types import Decision


# Default loss matrix [action, true_class]
# Rows: Green, Yellow, Red
# Cols: MCAR, MAR, MNAR
DEFAULT_LOSS_MATRIX = torch.tensor([
    [0.0,  0.0, 10.0],  # Green: safe for MCAR/MAR, costly for MNAR
    [1.0,  1.0,  2.0],  # Yellow: small cost always, less bad for MNAR
    [3.0,  2.0,  0.0],  # Red: costly for MCAR/MAR, free for MNAR
])


def compute_expected_loss(
    p_class: torch.Tensor,
    loss_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute expected loss for each action.
    
    Args:
        p_class: [B, 3] class posterior.
        loss_matrix: [3, 3] loss[action, class].
    
    Returns:
        [B, 3] expected loss for each action.
    """
    # E[L|a] = Σ_c L[a,c] * p(c)
    # [B, 3] = [B, 3] @ [3, 3].T
    return p_class @ loss_matrix.T


def bayes_optimal_decision(
    p_class: torch.Tensor,
    loss_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Bayes-optimal decision minimizing expected loss.
    
    Args:
        p_class: [B, 3] class posterior.
        loss_matrix: [3, 3] loss[action, true_class].
    
    Returns:
        action_ids: [B] optimal action indices (0=Green, 1=Yellow, 2=Red).
        expected_risks: [B] expected loss under optimal action.
    """
    # Ensure loss matrix is on same device
    loss_matrix = loss_matrix.to(p_class.device)
    
    # Compute expected loss for each action
    expected_loss = compute_expected_loss(p_class, loss_matrix)  # [B, 3]
    
    # Optimal action minimizes expected loss
    action_ids = expected_loss.argmin(dim=-1)  # [B]
    
    # Expected risk is the loss under optimal action
    expected_risks = expected_loss.gather(1, action_ids.unsqueeze(1)).squeeze(1)  # [B]
    
    return action_ids, expected_risks


def make_decision(
    p_class: torch.Tensor,
    loss_matrix: torch.Tensor = None,
) -> Decision:
    """Create Decision object from class posterior.
    
    Args:
        p_class: [B, 3] class posterior.
        loss_matrix: [3, 3] loss matrix. Uses default if None.
    
    Returns:
        Decision dataclass with actions and risks.
    """
    if loss_matrix is None:
        loss_matrix = DEFAULT_LOSS_MATRIX
    
    action_ids, expected_risks = bayes_optimal_decision(p_class, loss_matrix)
    
    return Decision(
        action_ids=action_ids,
        action_names=("Green", "Yellow", "Red"),
        expected_risks=expected_risks,
    )


def interpret_decision(decision: Decision, idx: int = 0) -> str:
    """Generate human-readable interpretation of a decision.
    
    Args:
        decision: Decision object.
        idx: Batch index to interpret.
    
    Returns:
        Human-readable string.
    """
    action = decision.get_actions()[idx]
    risk = decision.expected_risks[idx].item()
    
    interpretations = {
        "Green": "Proceed with MAR assumption. Low risk of MNAR bias.",
        "Yellow": "Proceed with caution. Consider sensitivity analysis.",
        "Red": "Do not assume MAR. Sensitivity analysis strongly recommended.",
    }
    
    return f"Decision: {action} (expected risk: {risk:.3f})"