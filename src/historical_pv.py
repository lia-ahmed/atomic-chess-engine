#!/usr/bin/env python3
"""Small, dependency-light helpers for historical policy+value training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def historical_value_target(result: int, side_to_move: int) -> float:
    """Convert game result to value from the stored side-to-move perspective.

    Args:
        result: +1 for White win, 0 for draw, -1 for Black win.
        side_to_move: 0 for White, 1 for Black.
    """
    result = int(result)
    side_to_move = int(side_to_move)
    if result not in (-1, 0, 1):
        raise ValueError(f"historical result must be -1, 0 or +1, got {result}")
    if side_to_move not in (0, 1):
        raise ValueError(f"side_to_move must be 0 (white) or 1 (black), got {side_to_move}")
    return float(result if side_to_move == 0 else -result)


def combined_loss(
    policy_logits: torch.Tensor,
    value_predictions: torch.Tensor,
    action_targets: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    policy_weight: float,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, policy cross-entropy, and value MSE losses."""
    policy_loss = F.cross_entropy(policy_logits, action_targets)
    value_loss = F.mse_loss(value_predictions, value_targets)
    total = policy_weight * policy_loss + value_weight * value_loss
    return total, policy_loss, value_loss
