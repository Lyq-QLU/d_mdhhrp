"""基于当前状态和插入评估生成专家监督标签。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - package/script dual use
    from .graph_builder import GraphData
    from .models import DMDHHRP_Environment, Patient, Route
except ImportError:  # pragma: no cover
    from graph_builder import GraphData
    from models import DMDHHRP_Environment, Patient, Route


@dataclass
class ExpertLabel:
    """一个决策时刻的专家监督信号。"""

    priority_node_idx: Optional[int]
    assignment_patient_node_idx: Optional[int]
    assignment_depot_node_idx: Optional[int]
    assignment_position: Optional[int]
    patient_node_indices: List[int] = field(default_factory=list)
    depot_node_indices: List[int] = field(default_factory=list)
    priority_soft_targets: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    assignment_soft_targets: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    patient_best_scores: Dict[int, float] = field(default_factory=dict)
    selected_patient_best_scores_by_depot: Dict[int, float] = field(default_factory=dict)
    # action_type: insert / wait / reject
    action_type: str = "insert"


def _candidate_routes(environment: DMDHHRP_Environment) -> List[Route]:
    routes = list(environment.routes)
    active_depot_ids = {route.depot.id for route in routes}
    for depot in environment.depots:
        if depot.id not in active_depot_ids:
            routes.append(Route(depot=depot))
    return routes


def _best_insertion_for_patient(
    environment: DMDHHRP_Environment,
    patient: Patient,
    routes: Sequence[Route],
    graph: GraphData,
    weight_cost: float,
    weight_sat: float,
) -> Tuple[float, Optional[int], Optional[int]]:
    best_score = float("inf")
    best_depot_node_idx: Optional[int] = None
    best_position: Optional[int] = None

    for route in routes:
        if patient.candidate_centers and route.depot.id not in patient.candidate_centers:
            continue

        for pos in range(len(route.patients) + 1):
            if pos < len(route.frozen_nodes):
                continue

            eval_res = environment.evaluate_insertion(route, patient, pos)
            if not eval_res.get("feasible", False):
                continue

            score = weight_cost * float(eval_res["delta_cost"]) - weight_sat * float(eval_res["delta_satisfaction"])
            if score < best_score:
                best_score = score
                best_position = pos
                best_depot_node_idx = graph.depot_id_to_node_idx.get(route.depot.id)

    return best_score, best_depot_node_idx, best_position


def generate_expert_label(
    environment: DMDHHRP_Environment,
    graph: Optional[GraphData] = None,
    *,
    weight_cost: float = 1.0,
    weight_sat: float = 100.0,
    temperature: float = 1.0,
) -> ExpertLabel:
    """基于插入评估生成状态依赖的专家标签。"""

    if graph is None:
        from .graph_builder import build_graph_from_env

        graph = build_graph_from_env(environment, include_served_patients=False)

    patient_node_indices = [int(idx) for idx in graph.patient_indices.tolist()]
    depot_node_indices = [int(idx) for idx in graph.depot_indices.tolist()]
    routes = _candidate_routes(environment)

    patient_best_scores: Dict[int, float] = {}
    patient_best_routes: Dict[int, Tuple[Optional[int], Optional[int]]] = {}

    for patient_node_idx in patient_node_indices:
        patient_obj = graph.node_objects[patient_node_idx]
        best_score, best_depot_node_idx, best_position = _best_insertion_for_patient(
            environment,
            patient_obj,
            routes,
            graph,
            weight_cost,
            weight_sat,
        )

        if best_score < float("inf"):
            patient_best_scores[patient_node_idx] = float(best_score)
            patient_best_routes[patient_node_idx] = (best_depot_node_idx, best_position)

    if not patient_best_scores:
        return ExpertLabel(
            priority_node_idx=None,
            assignment_patient_node_idx=None,
            assignment_depot_node_idx=None,
            assignment_position=None,
            patient_node_indices=patient_node_indices,
            depot_node_indices=depot_node_indices,
            action_type="wait",
        )

    priority_node_idx = min(patient_best_scores.items(), key=lambda item: item[1])[0]
    assignment_patient_node_idx = int(priority_node_idx)
    assignment_depot_node_idx, assignment_position = patient_best_routes[assignment_patient_node_idx]

    score_tensor = torch.full((len(patient_node_indices),), float("inf"), dtype=torch.float32)
    for local_idx, node_idx in enumerate(patient_node_indices):
        if node_idx in patient_best_scores:
            score_tensor[local_idx] = float(patient_best_scores[node_idx])

    finite_mask = torch.isfinite(score_tensor)
    priority_soft_targets = torch.zeros_like(score_tensor)
    if finite_mask.any():
        finite_scores = score_tensor[finite_mask]
        probs = torch.softmax(-finite_scores / max(float(temperature), 1e-6), dim=0)
        priority_soft_targets[finite_mask] = probs

    assignment_soft_targets = torch.zeros((len(depot_node_indices),), dtype=torch.float32)
    selected_scores_by_depot: Dict[int, float] = {}
    if assignment_patient_node_idx is not None:
        selected_patient = graph.node_objects[assignment_patient_node_idx]
        depot_scores: List[float] = []
        for depot_node_idx in depot_node_indices:
            depot_id = graph.node_idx_to_depot_id.get(depot_node_idx, -1)
            best_score = float("inf")
            for route in routes:
                if route.depot.id != depot_id:
                    continue
                if selected_patient.candidate_centers and route.depot.id not in selected_patient.candidate_centers:
                    continue
                for pos in range(len(route.patients) + 1):
                    if pos < len(route.frozen_nodes):
                        continue
                    eval_res = environment.evaluate_insertion(route, selected_patient, pos)
                    if not eval_res.get("feasible", False):
                        continue
                    score = weight_cost * float(eval_res["delta_cost"]) - weight_sat * float(eval_res["delta_satisfaction"])
                    best_score = min(best_score, score)

            if best_score < float("inf"):
                selected_scores_by_depot[depot_node_idx] = best_score
                depot_scores.append(best_score)
            else:
                depot_scores.append(float("inf"))

        depot_scores_tensor = torch.tensor(depot_scores, dtype=torch.float32)
        finite_depot_mask = torch.isfinite(depot_scores_tensor)
        if finite_depot_mask.any():
            probs = torch.softmax(-depot_scores_tensor[finite_depot_mask] / max(float(temperature), 1e-6), dim=0)
            assignment_soft_targets[finite_depot_mask] = probs

    return ExpertLabel(
        priority_node_idx=int(priority_node_idx),
        assignment_patient_node_idx=assignment_patient_node_idx,
        assignment_depot_node_idx=assignment_depot_node_idx,
        assignment_position=assignment_position,
        patient_node_indices=patient_node_indices,
        depot_node_indices=depot_node_indices,
        priority_soft_targets=priority_soft_targets,
        assignment_soft_targets=assignment_soft_targets,
        patient_best_scores=patient_best_scores,
        selected_patient_best_scores_by_depot=selected_scores_by_depot,
    )


def apply_expert_label(environment, graph: GraphData, label: ExpertLabel) -> bool:
    """Apply an expert insert action to the environment based on the provided label.

    Returns True if an insertion was applied successfully, False otherwise.
    """
    if label.action_type != "insert":
        return False

    if label.assignment_patient_node_idx is None or label.assignment_depot_node_idx is None or label.assignment_position is None:
        return False

    patient = graph.node_objects[label.assignment_patient_node_idx]
    depot_id = graph.node_idx_to_depot_id.get(label.assignment_depot_node_idx)

    target_route = None
    for route in environment.routes:
        if route.depot.id == depot_id:
            target_route = route
            break

    if target_route is None:
        for depot in environment.depots:
            if depot.id == depot_id:
                new_route = Route(depot)
                environment.routes.append(new_route)
                target_route = new_route
                break

    if target_route is None:
        return False

    pos = int(label.assignment_position)
    eval_res = environment.evaluate_insertion(target_route, patient, pos)
    if not eval_res.get("feasible", False):
        return False

    target_route.patients.insert(pos, patient)
    # simulate and sync planning state
    feasible, start_times = environment.simulate_route(target_route)
    if not feasible:
        # rollback
        target_route.patients.pop(pos)
        return False

    target_route.service_start_times = start_times
    target_route.sync_planning_state()
    patient.is_served = True
    return True


__all__ = ["ExpertLabel", "generate_expert_label", "apply_expert_label"]