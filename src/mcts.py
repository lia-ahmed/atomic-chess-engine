#!/usr/bin/env python3
"""PUCT Monte Carlo Tree Search for atomic chess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
import torch

import chess
import chess.variant

from representations import board_to_planes, normalize_uci


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(frozen=True)
class SearchResult:
    move_uci: str
    visit_counts: Dict[str, int]
    visit_probs: Dict[str, float]
    root_priors: Dict[str, float]
    root_value: float
    simulations: int


class MCTS:
    def __init__(
        self,
        net: torch.nn.Module,
        action_map: Mapping[str, int],
        *,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        device: str | torch.device | None = None,
        dirichlet_alpha: float = 0.0,
        dirichlet_epsilon: float = 0.0,
        unknown_move_prior_mass: float = 0.0,
        seed: int = 0,
    ) -> None:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if not 0.0 <= dirichlet_epsilon <= 1.0:
            raise ValueError("dirichlet_epsilon must be in [0,1]")
        if dirichlet_epsilon > 0.0 and dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive when noise is enabled")
        if not 0.0 <= unknown_move_prior_mass < 1.0:
            raise ValueError("unknown_move_prior_mass must be in [0,1)")

        self.net = net
        self.action_map = {normalize_uci(k): int(v) for k, v in action_map.items()}
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.unknown_move_prior_mass = float(unknown_move_prior_mass)
        self.rng = np.random.default_rng(seed)

        if device is None:
            try:
                self.device = next(net.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.net.eval()

    def search(
        self,
        state: str | chess.variant.AtomicBoard,
        *,
        temperature: float = 0.0,
        add_root_noise: bool = False,
    ) -> SearchResult:
        board = _coerce_board(state)
        if _terminal_outcome(board) is not None:
            raise ValueError("cannot search a terminal position")

        root = Node(prior=1.0)
        root_value = self._evaluate_and_expand(root, board)
        if not root.children:
            raise ValueError("position has no legal moves")

        if add_root_noise and self.dirichlet_epsilon > 0.0:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            sim_board = board.copy(stack=False)
            node = root
            path: list[Node] = []

            while node.expanded and node.children:
                move_uci, child = self._select_child(node)
                move = chess.Move.from_uci(move_uci)
                # children are created only from legal moves in this exact position.
                sim_board.push(move)
                path.append(child)
                node = child

                outcome = _terminal_outcome(sim_board)
                if outcome is not None:
                    leaf_value = _terminal_value_for_side_to_move(sim_board, outcome)
                    break
            else:
                outcome = _terminal_outcome(sim_board)
                if outcome is not None:
                    leaf_value = _terminal_value_for_side_to_move(sim_board, outcome)
                else:
                    leaf_value = self._evaluate_and_expand(node, sim_board)

            # leaf_value is for the player to move at the leaf. Each parent is the
            # opponent, so negate before updating the incoming edge statistics.
            value = float(leaf_value)
            for child in reversed(path):
                value = -value
                child.visit_count += 1
                child.value_sum += value

        visit_counts = {uci: child.visit_count for uci, child in root.children.items()}
        visit_probs = _counts_to_probs(visit_counts, temperature=1.0)
        selected = _sample_from_counts(visit_counts, temperature, self.rng)
        priors = {uci: child.prior for uci, child in root.children.items()}
        return SearchResult(
            move_uci=selected,
            visit_counts=visit_counts,
            visit_probs=visit_probs,
            root_priors=priors,
            root_value=float(root_value),
            simulations=self.num_simulations,
        )

    @torch.no_grad()
    def _evaluate_and_expand(self, node: Node, board: chess.variant.AtomicBoard) -> float:
        outcome = _terminal_outcome(board)
        if outcome is not None:
            node.expanded = True
            return _terminal_value_for_side_to_move(board, outcome)

        planes = board_to_planes(board)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device, dtype=torch.float32)
        output = self.net(tensor)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError("Network must return (policy_logits, value)")
        logits, value = output
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"expected policy logits [1,A], got {tuple(logits.shape)}")
        if value.numel() != 1:
            raise ValueError(f"expected scalar value, got shape {tuple(value.shape)}")

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            node.expanded = True
            return float(value.reshape(-1)[0].item())

        priors = self._legal_priors(logits[0], legal_moves)
        node.children = {move.uci(): Node(prior=float(priors[move.uci()])) for move in legal_moves}
        node.expanded = True
        return float(value.reshape(-1)[0].clamp(-1, 1).item())

    def _legal_priors(
        self,
        logits: torch.Tensor,
        legal_moves: list[chess.Move],
    ) -> Dict[str, float]:
        known_uci: list[str] = []
        known_indices: list[int] = []
        unknown_uci: list[str] = []

        for move in legal_moves:
            uci = move.uci()
            idx = self.action_map.get(uci)
            if idx is None:
                unknown_uci.append(uci)
            else:
                if idx < 0 or idx >= logits.shape[0]:
                    raise ValueError(f"action index {idx} for {uci} exceeds network output size")
                known_uci.append(uci)
                known_indices.append(idx)

        if not known_uci:
            p = 1.0 / len(legal_moves)
            return {move.uci(): p for move in legal_moves}

        idx_tensor = torch.tensor(known_indices, device=logits.device, dtype=torch.long)
        known_logits = logits.index_select(0, idx_tensor)
        known_probs = torch.softmax(known_logits.float(), dim=0).detach().cpu().numpy()

        unknown_mass = self.unknown_move_prior_mass if unknown_uci else 0.0
        known_mass = 1.0 - unknown_mass
        priors = {
            uci: float(prob * known_mass)
            for uci, prob in zip(known_uci, known_probs.tolist())
        }
        if unknown_uci:
            each = unknown_mass / len(unknown_uci)
            priors.update({uci: each for uci in unknown_uci})

        total = sum(priors.values())
        if total <= 0.0 or not math.isfinite(total):
            p = 1.0 / len(legal_moves)
            return {move.uci(): p for move in legal_moves}
        return {uci: p / total for uci, p in priors.items()}

    def _select_child(self, node: Node) -> tuple[str, Node]:
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(max(1, total_visits))

        best_score = -math.inf
        best: list[tuple[str, Node]] = []
        for uci, child in node.children.items():
            exploration = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = child.q_value + exploration
            if score > best_score + 1e-12:
                best_score = score
                best = [(uci, child)]
            elif abs(score - best_score) <= 1e-12:
                best.append((uci, child))
        if not best:
            raise RuntimeError("PUCT selection found no child")
        return best[int(self.rng.integers(len(best)))]

    def _add_dirichlet_noise(self, root: Node) -> None:
        keys = list(root.children)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(keys))
        eps = self.dirichlet_epsilon
        for uci, eta in zip(keys, noise.tolist()):
            child = root.children[uci]
            child.prior = (1.0 - eps) * child.prior + eps * float(eta)


def _coerce_board(state: str | chess.variant.AtomicBoard) -> chess.variant.AtomicBoard:
    if isinstance(state, chess.variant.AtomicBoard):
        return state.copy(stack=False)
    if isinstance(state, str):
        return chess.variant.AtomicBoard(state)
    raise TypeError("state must be an AtomicBoard or FEN string")


def _terminal_outcome(board: chess.variant.AtomicBoard):
    return board.outcome(claim_draw=True)


def _terminal_value_for_side_to_move(board: chess.variant.AtomicBoard, outcome) -> float:
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _counts_to_probs(counts: Mapping[str, int], temperature: float) -> Dict[str, float]:
    if not counts:
        return {}
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    keys = list(counts)
    values = np.asarray([max(0, int(counts[k])) for k in keys], dtype=np.float64)
    if values.sum() <= 0:
        values = np.ones_like(values)
    if temperature == 0.0:
        probs = np.zeros_like(values)
        probs[int(np.argmax(values))] = 1.0
    else:
        scaled = np.power(values, 1.0 / temperature)
        if scaled.sum() <= 0 or not np.isfinite(scaled).all():
            scaled = np.ones_like(values)
        probs = scaled / scaled.sum()
    return {k: float(p) for k, p in zip(keys, probs.tolist())}


def _sample_from_counts(
    counts: Mapping[str, int],
    temperature: float,
    rng: np.random.Generator,
) -> str:
    probs = _counts_to_probs(counts, temperature)
    keys = list(probs)
    p = np.asarray([probs[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=p))]


def select_action(
    state: str | chess.variant.AtomicBoard,
    net: torch.nn.Module,
    num_simulations: int,
    c_puct: float,
    *,
    action_map: Optional[Mapping[str, int]] = None,
    temperature: float = 0.0,
    seed: int = 0,
) -> str:
    """Plan-compatible convenience wrapper returning one UCI move.

    'action_map' may be passed explicitly or attached to 'net.action_map'.
    """

    if action_map is None:
        action_map = getattr(net, "action_map", None)
    if action_map is None:
        raise ValueError("action_map is required (argument or net.action_map)")
    searcher = MCTS(
        net,
        action_map,
        num_simulations=num_simulations,
        c_puct=c_puct,
        seed=seed,
    )
    return searcher.search(state, temperature=temperature).move_uci
