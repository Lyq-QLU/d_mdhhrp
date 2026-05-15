"""基于专家标签的 GNN 监督训练入口。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:  # pragma: no cover - package/script dual use
    from .data_loader import build_environment, load_hybrid_instance_from_json
    from .graph_builder import build_graph_from_env
    from .gnn_policy import GNNDispatchPolicy
    from .label_generator import generate_expert_label, apply_expert_label
except ImportError:  # pragma: no cover
    from data_loader import build_environment, load_hybrid_instance_from_json
    from graph_builder import build_graph_from_env
    from gnn_policy import GNNDispatchPolicy
    from label_generator import generate_expert_label, apply_expert_label


@dataclass
class TrainConfig:
    dataset_dir: str = "/Users/marc/Documents/Code/dataset/main"
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_cost: float = 1.0
    weight_sat: float = 100.0
    lambda_assign: float = 1.0
    kl_weight: float = 0.1
    temperature: float = 1.0
    limit: Optional[int] = None
    device: str = "cpu"


def _collect_instance_files(dataset_dir: str, limit: Optional[int] = None) -> List[Path]:
    root = Path(dataset_dir)
    files = [p for p in sorted(root.rglob("*.json")) if p.name != "metadata.json"]
    if limit is not None:
        files = files[:limit]
    return files


def _iter_decision_times(environment) -> List[float]:
    event_times = {float(environment.current_time)}
    for patient in environment.dynamic_patients_pool:
        event_times.add(float(patient.arrival_time))
    return sorted(event_times)


def _local_index_from_global(index_tensor: torch.Tensor, global_idx: int) -> Optional[int]:
    matches = torch.nonzero(index_tensor == int(global_idx), as_tuple=False)
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


def _train_one_state(
    policy: GNNDispatchPolicy,
    optimizer: torch.optim.Optimizer,
    environment,
    *,
    weight_cost: float,
    weight_sat: float,
    lambda_assign: float,
    kl_weight: float,
    temperature: float,
) -> Optional[Dict[str, float]]:
    device = next(policy.parameters()).device
    graph = build_graph_from_env(environment, include_served_patients=False).to(device)
    label = generate_expert_label(
        environment,
        graph,
        weight_cost=weight_cost,
        weight_sat=weight_sat,
        temperature=temperature,
    )

    # If no feasible insertion -> return label for caller to record and optionally skip applying
    if label.priority_node_idx is None or label.assignment_patient_node_idx is None or label.assignment_depot_node_idx is None:
        return None, label

    priority_local = _local_index_from_global(graph.patient_indices, label.priority_node_idx)
    assignment_patient_local = _local_index_from_global(graph.patient_indices, label.assignment_patient_node_idx)
    assignment_depot_local = _local_index_from_global(graph.depot_indices, label.assignment_depot_node_idx)
    if priority_local is None or assignment_patient_local is None or assignment_depot_local is None:
        return None

    output = policy(graph)

    loss_priority = F.cross_entropy(
        output.priority_logits.unsqueeze(0),
        torch.tensor([priority_local], device=device),
    )

    if output.assignment_logits.numel() == 0:
        return None
    assignment_row = output.assignment_logits[assignment_patient_local].unsqueeze(0)
    loss_assign = F.cross_entropy(
        assignment_row,
        torch.tensor([assignment_depot_local], device=device),
    )

    loss = loss_priority + lambda_assign * loss_assign

    kl_priority = torch.tensor(0.0, device=device)
    if label.priority_soft_targets.numel() == output.priority_logits.numel() and float(label.priority_soft_targets.sum()) > 0:
        target = label.priority_soft_targets.to(device)
        kl_priority = F.kl_div(F.log_softmax(output.priority_logits, dim=-1), target, reduction="batchmean")
        loss = loss + kl_weight * kl_priority

    kl_assign = torch.tensor(0.0, device=device)
    if label.assignment_soft_targets.numel() == assignment_row.numel() and float(label.assignment_soft_targets.sum()) > 0:
        target = label.assignment_soft_targets.to(device)
        kl_assign = F.kl_div(F.log_softmax(assignment_row.squeeze(0), dim=-1), target, reduction="batchmean")
        loss = loss + kl_weight * kl_assign

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (
        {
            "loss": float(loss.item()),
            "loss_priority": float(loss_priority.item()),
            "loss_assign": float(loss_assign.item()),
            "kl_priority": float(kl_priority.item()),
            "kl_assign": float(kl_assign.item()),
        },
        label,
    )


def train_supervised_gnn(config: TrainConfig) -> Tuple[GNNDispatchPolicy, List[Dict[str, float]]]:
    """用专家标签训练 GNNDispatchPolicy。"""

    device = torch.device(config.device)
    policy = GNNDispatchPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    instance_files = _collect_instance_files(config.dataset_dir, config.limit)
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_steps = 0
        skipped_no_feasible_label_count = 0
        applied_expert_count = 0
        failed_apply_count = 0

        for file_path in instance_files:
            instance = load_hybrid_instance_from_json(file_path)
            environment = build_environment(instance)

            for current_time in _iter_decision_times(environment):
                environment.update_time(current_time)
                environment.freeze_prefixes(current_time)

                metrics, label = _train_one_state(
                    policy,
                    optimizer,
                    environment,
                    weight_cost=config.weight_cost,
                    weight_sat=config.weight_sat,
                    lambda_assign=config.lambda_assign,
                    kl_weight=config.kl_weight,
                    temperature=config.temperature,
                )

                if label is None:
                    # generation failed unexpectedly; skip
                    continue

                if metrics is None:
                    # no feasible insertion at this state
                    skipped_no_feasible_label_count += 1
                    # still try applying (will be a no-op) for completeness
                    try:
                        applied = False
                        # apply_expert_label will return False for wait/reject
                        applied = apply_expert_label(environment, build_graph_from_env(environment, include_served_patients=False).to(next(policy.parameters()).device, ), label)
                        if applied:
                            applied_expert_count += 1
                        else:
                            failed_apply_count += 1
                    except Exception:
                        failed_apply_count += 1
                    continue

                epoch_loss += metrics["loss"]
                epoch_steps += 1

                # Attempt to apply the expert action to advance the environment state
                try:
                    applied = apply_expert_label(environment, build_graph_from_env(environment, include_served_patients=False), label)
                    if applied:
                        applied_expert_count += 1
                    else:
                        failed_apply_count += 1
                except Exception:
                    failed_apply_count += 1

        history.append(
            {
                "epoch": float(epoch + 1),
                "avg_loss": float(epoch_loss / max(epoch_steps, 1)),
                "steps": float(epoch_steps),
                "skipped_no_feasible_label_count": int(skipped_no_feasible_label_count),
                "applied_expert_count": int(applied_expert_count),
                "failed_apply_count": int(failed_apply_count),
            }
        )

    return policy, history


def main(
    dataset_dir: str = "/Users/marc/Documents/Code/dataset/main",
    epochs: int = 3,
    learning_rate: float = 1e-3,
    limit: Optional[int] = None,
):
    config = TrainConfig(dataset_dir=dataset_dir, epochs=epochs, learning_rate=learning_rate, limit=limit)
    solver, history = train_supervised_gnn(config)
    print("training_history:")
    for item in history:
        print(item)
    return solver, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于专家标签训练 GNNDispatchPolicy")
    parser.add_argument("--dataset-dir", type=str, default="/Users/marc/Documents/Code/dataset/main")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        limit=args.limit,
    )
