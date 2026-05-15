"""GNN 前向推理后接 Greedy / 轻量修复调度。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

try:  # pragma: no cover - package/script dual use
    from .graph_builder import GraphData, build_graph_from_env
    from .gnn_policy import GNNDispatchPolicy
    from .models import DMDHHRP_Environment, DynamicPatient, Solution
    from .operators import (
        CenterAwareRegretRepair,
        CenterCloseRemoval,
        GreedyRepair,
        HighImpactCenterRemoval,
        RandomRemoval,
        Regret2Repair,
        SatisfactionAwareRepair,
        ShawRemoval,
        TimeWindowFirstRepair,
        TimeWindowViolationRemoval,
        WorstRemoval,
        WorstSatisfactionRemoval,
    )
    from .policy import OperatorSelectorPolicy, build_state_vector, dominates
except ImportError:  # pragma: no cover
    from graph_builder import GraphData, build_graph_from_env
    from gnn_policy import GNNDispatchPolicy
    from models import DMDHHRP_Environment, DynamicPatient, Solution
    from operators import (
        CenterAwareRegretRepair,
        CenterCloseRemoval,
        GreedyRepair,
        HighImpactCenterRemoval,
        RandomRemoval,
        Regret2Repair,
        SatisfactionAwareRepair,
        ShawRemoval,
        TimeWindowFirstRepair,
        TimeWindowViolationRemoval,
        WorstRemoval,
        WorstSatisfactionRemoval,
    )
    from policy import OperatorSelectorPolicy, build_state_vector, dominates


@dataclass
class DispatchResult:
    solution: Solution
    accepted_dynamic_ids: List[int] = field(default_factory=list)
    pending_dynamic_ids: List[int] = field(default_factory=list)
    patient_priority_order: List[int] = field(default_factory=list)
    top_center_choices: Dict[int, List[int]] = field(default_factory=dict)
    alns_improved: bool = False
    alns_rounds: int = 0


def _is_patient(node_obj: object) -> bool:
    return hasattr(node_obj, "is_dynamic") or hasattr(node_obj, "hard_tw")


def _patient_node_indices(graph: GraphData) -> List[int]:
    return graph.patient_indices.tolist()


def _depot_node_indices(graph: GraphData) -> List[int]:
    return graph.depot_indices.tolist()


def _priority_order(graph: GraphData, priority_logits: torch.Tensor) -> List[int]:
    if priority_logits.numel() == 0:
        return []
    order = torch.argsort(priority_logits, descending=True)
    return [int(graph.patient_indices[idx]) for idx in order.tolist()]


def _center_scores_by_patient(graph: GraphData, assignment_output: torch.Tensor) -> Dict[int, List[tuple[int, float]]]:
    scores_by_patient: Dict[int, List[tuple[int, float]]] = {}
    if assignment_output.numel() == 0:
        return scores_by_patient
    if assignment_output.dim() == 2:
        patient_nodes = graph.patient_indices.tolist()
        depot_nodes = graph.depot_indices.tolist()
        for p_local, patient_node_idx in enumerate(patient_nodes):
            for d_local, depot_node_idx in enumerate(depot_nodes):
                score = float(assignment_output[p_local, d_local].item())
                if score <= -1e8:
                    continue
                scores_by_patient.setdefault(int(patient_node_idx), []).append((int(depot_node_idx), score))
    else:
        for idx, (patient_idx, depot_idx) in enumerate(graph.pair_index.tolist()):
            scores_by_patient.setdefault(int(patient_idx), []).append((int(depot_idx), float(assignment_output[idx].item())))

    for patient_idx, items in scores_by_patient.items():
        items.sort(key=lambda item: item[1], reverse=True)
    return scores_by_patient


def _apply_candidate_centers(graph: GraphData, patient_node_idx: int, depot_node_indices: Sequence[int], top_k: int) -> List[int]:
    patient_obj = graph.node_objects[patient_node_idx]
    if not _is_patient(patient_obj):
        return []
    top_k = max(1, int(top_k))
    candidate_ids: List[int] = []
    for depot_node_idx in depot_node_indices[:top_k]:
        depot_obj = graph.node_objects[depot_node_idx]
        candidate_ids.append(int(getattr(depot_obj, "id")))
    patient_obj.candidate_centers = candidate_ids
    return candidate_ids


def _scalar_score(solution: Solution) -> float:
    return float(solution.obj1) - 100.0 * float(solution.obj2)


def _capture_environment(environment: DMDHHRP_Environment) -> DMDHHRP_Environment:
    return copy.deepcopy(environment)


def _restore_environment(environment: DMDHHRP_Environment, snapshot: DMDHHRP_Environment) -> None:
    environment.__dict__.clear()
    environment.__dict__.update(copy.deepcopy(snapshot.__dict__))


def _run_light_alns(
    environment: DMDHHRP_Environment,
    rounds: int,
    destroy_ratio: float,
    deterministic_policy: bool = True,
) -> bool:
    if rounds <= 0:
        return False

    policy = OperatorSelectorPolicy(seed=7)
    destroy_ops = [
        RandomRemoval(),
        ShawRemoval(),
        WorstRemoval(),
        WorstSatisfactionRemoval(),
        TimeWindowViolationRemoval(),
        HighImpactCenterRemoval(),
        CenterCloseRemoval(),
    ]
    repair_ops = [
        GreedyRepair(),
        Regret2Repair(),
        SatisfactionAwareRepair(),
        TimeWindowFirstRepair(),
        CenterAwareRegretRepair(),
        GreedyRepair(),
    ]

    best_snapshot = _capture_environment(environment)
    best_solution = environment._export_solution()
    best_solution.obj1, best_solution.obj2 = environment.evaluate_objectives()
    best_score = _scalar_score(best_solution)
    improved = False

    for round_idx in range(rounds):
        current_solution = environment._export_solution()
        current_solution.obj1, current_solution.obj2 = environment.evaluate_objectives()
        state = build_state_vector(
            current_solution,
            search_history={"iteration": round_idx, "max_iterations": rounds, "best_cost": best_solution.obj1, "best_satisfaction": best_solution.obj2},
            waiting_dynamic_count=len([p for p in environment.known_patients if p.is_dynamic and not p.is_served]),
            life_circle_coverage=0.0,
            rejection_ratio=len(environment.rejected_dynamic) / max(len(environment.known_patients), 1),
        )
        action, _, _ = policy.select_action(state, deterministic=deterministic_policy)
        destroy_idx, repair_idx = policy.decode_action(action)
        destroy_idx %= len(destroy_ops)
        repair_idx %= len(repair_ops)

        round_snapshot = _capture_environment(environment)
        removed_patients = destroy_ops[destroy_idx].apply(environment, destroy_ratio=destroy_ratio)
        if removed_patients:
            repair_ops[repair_idx].apply(environment, removed_patients)

        candidate_solution = environment._export_solution()
        candidate_solution.obj1, candidate_solution.obj2 = environment.evaluate_objectives()
        candidate_score = _scalar_score(candidate_solution)

        if candidate_score + 1e-9 < best_score or dominates(candidate_solution, best_solution):
            best_snapshot = _capture_environment(environment)
            best_solution = candidate_solution
            best_score = candidate_score
            improved = True
        else:
            _restore_environment(environment, round_snapshot)

    _restore_environment(environment, best_snapshot)
    environment.evaluate_objectives()
    return improved


def dispatch_dynamic_patients(
    environment: DMDHHRP_Environment,
    policy: Optional[GNNDispatchPolicy] = None,
    waiting_dynamic: Optional[Sequence[DynamicPatient]] = None,
    top_k_centers: int = 2,
    use_greedy_fallback: bool = True,
    include_served_patients: bool = False,
    alns_rounds: int = 0,
    alns_destroy_ratio: float = 0.2,
) -> DispatchResult:
    """执行一次动态图构建 + GNN 推理 + greedy/repair 插入。"""

    if policy is None:
        policy = GNNDispatchPolicy()
    policy.eval()

    if waiting_dynamic is None:
        waiting_pool = [patient for patient in environment.known_patients if patient.is_dynamic and not patient.is_served]
    else:
        waiting_pool = [patient for patient in waiting_dynamic if not patient.is_served]

    graph = build_graph_from_env(
        environment,
        include_served_patients=include_served_patients,
        waiting_dynamic=waiting_pool,
    )
    with torch.no_grad():
        output = policy.predict(graph)

    patient_priority_order = _priority_order(graph, output["priority_logits"])
    assignment_output = output.get("assignment_logits", output["assignment_scores"])
    scores_by_patient = _center_scores_by_patient(graph, assignment_output)

    waiting_ids = {patient.id for patient in waiting_pool}
    accepted_dynamic_ids: List[int] = []
    pending_dynamic_ids: List[int] = []
    top_center_choices: Dict[int, List[int]] = {}
    actual_priority_order: List[int] = []

    # 先按 GNN 优先级排序，再执行插入。
    ordered_waiting: List[DynamicPatient] = []
    for node_idx in patient_priority_order:
        patient_obj = graph.node_objects[node_idx]
        if getattr(patient_obj, "id", None) in waiting_ids:
            ordered_waiting.append(patient_obj)
            actual_priority_order.append(int(patient_obj.id))
    # 没被排序到的等待患者补在后面。
    for patient in waiting_pool:
        if patient not in ordered_waiting:
            ordered_waiting.append(patient)
            actual_priority_order.append(int(patient.id))

    for patient in ordered_waiting:
        original_centers = list(patient.candidate_centers)

        patient_idx = graph.patient_id_to_node_idx.get(patient.id)
        if patient_idx is None:
            continue

        candidate_pairs = scores_by_patient.get(patient_idx, [])
        candidate_center_ids: List[int] = []
        if candidate_pairs:
            depot_node_indices = [depot_idx for depot_idx, _ in candidate_pairs]
            candidate_center_ids = _apply_candidate_centers(graph, patient_idx, depot_node_indices, top_k_centers)
        else:
            candidate_center_ids = list(original_centers) if original_centers else [depot.id for depot in environment.depots]
            patient.candidate_centers = candidate_center_ids

        top_center_choices[patient.id] = list(candidate_center_ids)

        success = environment.greedy_insert(patient)
        if not success and use_greedy_fallback:
            patient.candidate_centers = list(original_centers) if original_centers else [depot.id for depot in environment.depots]
            success = environment.greedy_insert(patient)

        if success:
            patient.is_served = True
            accepted_dynamic_ids.append(patient.id)
        else:
            patient.candidate_centers = original_centers
            pending_dynamic_ids.append(patient.id)

    alns_improved = _run_light_alns(
        environment,
        rounds=max(0, int(alns_rounds)),
        destroy_ratio=max(0.05, min(0.8, float(alns_destroy_ratio))),
    )

    obj1, obj2 = environment.evaluate_objectives()
    solution = environment._export_solution()
    solution.obj1 = obj1
    solution.obj2 = obj2

    return DispatchResult(
        solution=solution,
        accepted_dynamic_ids=accepted_dynamic_ids,
        pending_dynamic_ids=pending_dynamic_ids,
        patient_priority_order=actual_priority_order,
        top_center_choices=top_center_choices,
        alns_improved=alns_improved,
        alns_rounds=max(0, int(alns_rounds)),
    )


__all__ = ["DispatchResult", "dispatch_dynamic_patients"]