#!/usr/bin/env python3
"""Neural-network models.

Value head returns a scalar in [-1, 1] from perspective of the side to move in the input position.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Policy-only model configuration."""

    num_actions: int
    in_channels: int = 14
    width: int = 64
    blocks: int = 6
    policy_channels: int = 32
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyValueConfig:
    """Policy+value model configuration."""

    num_actions: int
    in_channels: int = 14
    width: int = 64
    blocks: int = 6
    policy_channels: int = 32
    value_channels: int = 32
    value_hidden: int = 128
    dropout: float = 0.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class MovePolicyNet(nn.Module):
    """Residual CNN producing logits over the UCI action vocabulary."""

    def __init__(
        self,
        num_actions: int,
        *,
        in_channels: int = 14,
        width: int = 64,
        blocks: int = 6,
        policy_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dimensions(num_actions, in_channels, width, blocks, policy_channels, dropout)

        self.config = ModelConfig(
            num_actions=int(num_actions),
            in_channels=int(in_channels),
            width=int(width),
            blocks=int(blocks),
            policy_channels=int(policy_channels),
            dropout=float(dropout),
        )

        self.stem = _make_stem(in_channels, width)
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.policy_head = _make_policy_head(width, policy_channels, num_actions, dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _reset_module_parameters(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, self.config.in_channels)
        x = self.stem(x)
        x = self.body(x)
        return self.policy_head(x)


class PolicyValueNet(nn.Module):
    """Residual network with policy logits and a scalar value head.

    The value is always interpreted from the perspective of the side to move.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        in_channels: int = 14,
        width: int = 64,
        blocks: int = 6,
        policy_channels: int = 32,
        value_channels: int = 32,
        value_hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dimensions(num_actions, in_channels, width, blocks, policy_channels, dropout)
        if value_channels <= 0 or value_hidden <= 0:
            raise ValueError("value_channels and value_hidden must be positive")

        self.config = PolicyValueConfig(
            num_actions=int(num_actions),
            in_channels=int(in_channels),
            width=int(width),
            blocks=int(blocks),
            policy_channels=int(policy_channels),
            value_channels=int(value_channels),
            value_hidden=int(value_hidden),
            dropout=float(dropout),
        )

        # Keep these names identical to MovePolicyNet so Phase 3 weights copy directly.
        self.stem = _make_stem(in_channels, width)
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.policy_head = _make_policy_head(width, policy_channels, num_actions, dropout)
        self.value_head = nn.Sequential(
            nn.Conv2d(width, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_channels * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )
        _reset_module_parameters(self)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_input(x, self.config.in_channels)
        features = self.body(self.stem(x))
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value

    def zero_value_output(self) -> None:
        """Make the initial value prediction exactly zero for every position.

        This is useful immediately after transferring a policy-only checkpoint: 
        the first self-play iteration is then guided by policy priors and terminal outcomes instead of arbitrary random value estimates.
        """

        final_linear = self.value_head[-2]
        if not isinstance(final_linear, nn.Linear):
            raise RuntimeError("unexpected value-head layout")
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)


def _validate_dimensions(
    num_actions: int,
    in_channels: int,
    width: int,
    blocks: int,
    policy_channels: int,
    dropout: float,
) -> None:
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if in_channels <= 0 or width <= 0 or blocks < 0 or policy_channels <= 0:
        raise ValueError("invalid model dimensions")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")


def _make_stem(in_channels: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    )


def _make_policy_head(
    width: int,
    policy_channels: int,
    num_actions: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(width, policy_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(policy_channels),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(policy_channels * 8 * 8, num_actions),
    )


def _reset_module_parameters(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def _validate_input(x: torch.Tensor, in_channels: int) -> None:
    if x.ndim != 4 or x.shape[1:] != (in_channels, 8, 8):
        raise ValueError(
            f"expected input shape [batch,{in_channels},8,8], got {tuple(x.shape)}"
        )


def build_model(config: ModelConfig | Mapping[str, Any]) -> MovePolicyNet:
    if isinstance(config, ModelConfig):
        kwargs = asdict(config)
    else:
        kwargs = dict(config)
    return MovePolicyNet(**kwargs)


def build_policy_value_model(
    config: PolicyValueConfig | Mapping[str, Any],
) -> PolicyValueNet:
    if isinstance(config, PolicyValueConfig):
        kwargs = asdict(config)
    else:
        kwargs = dict(config)
    return PolicyValueNet(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def action_map_digest(action_map: Mapping[str, int]) -> str:
    payload = json.dumps(
        {str(k): int(v) for k, v in sorted(action_map.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def transfer_supervised_weights(
    model: PolicyValueNet,
    supervised_checkpoint: str | Path | Mapping[str, Any],
    *,
    zero_value: bool = True,
) -> dict[str, Any]:
    """Copy trunk/policy weights into 4 model.

    Returns the loaded checkpoint dictionary. The architecture must match for every transferred tensor.
    """

    if isinstance(supervised_checkpoint, (str, Path)):
        checkpoint = torch.load(supervised_checkpoint, map_location="cpu")
    else:
        checkpoint = dict(supervised_checkpoint)

    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("supervised checkpoint does not contain model_state")

    transferable = {
        key: value
        for key, value in state.items()
        if key.startswith("stem.") or key.startswith("body.") or key.startswith("policy_head.")
    }
    if not transferable:
        raise ValueError("no transferable weights found")

    current = model.state_dict()
    mismatched = [
        key
        for key, value in transferable.items()
        if key not in current or tuple(current[key].shape) != tuple(value.shape)
    ]
    if mismatched:
        raise ValueError(
            "Checkpoint is incompatible with requested  architecture; "
            f"mismatched keys include: {mismatched[:8]}"
        )

    missing, unexpected = model.load_state_dict(transferable, strict=False)
    if unexpected:
        raise ValueError(f"unexpected transferred keys: {unexpected}")
    non_value_missing = [key for key in missing if not key.startswith("value_head.")]
    if non_value_missing:
        raise ValueError(f"supervised transfer left non-value keys missing: {non_value_missing[:8]}")

    if zero_value:
        model.zero_value_output()
    return checkpoint


def load_policy_value_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PolicyValueNet, dict[str, Any]]:
    """Load a checkpoint and return '(model, checkpoint_dict)'."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint does not contain model_config")
    required = {"num_actions", "width", "blocks", "policy_channels"}
    if not required.issubset(config):
        raise ValueError(
            "checkpoint appears to be policy-only; create a Phase 4 checkpoint with "
            "init_phase4.py first"
        )
    if "value_channels" not in config or "value_hidden" not in config:
        raise ValueError(
            "checkpoint has no value-head configuration; create a Phase 4 checkpoint "
            "with init_phase4.py first"
        )
    model = PolicyValueNet(**dict(config)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    model = PolicyValueNet(num_actions=512, width=32, blocks=2)
    x = torch.zeros(4, 14, 8, 8)
    policy, value = model(x)
    print(
        f"policy_shape={tuple(policy.shape)} value_shape={tuple(value.shape)} "
        f"trainable_parameters={count_parameters(model):,}"
    )
