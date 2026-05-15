"""把当前环境构造成 GNN 可消费的动态图输入。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - package/script dual use
    from .models import DMDHHRP_Environment, Depot, DynamicPatient, Patient, Route
except ImportError:  # pragma: no cover
    from models import DMDHHRP_Environment, Depot, DynamicPatient, Patient, Route


def _distance_and_travel_time(a, b, speed: float) -> Tuple[float, float]:
    distance = a.location.distance_to(b.location)
    travel_time = distance / max(speed, 1e-9)
    return distance, travel_time


def _normalize(value: float, scale: float) -> float:
    return float(value) / max(float(scale), 1e-9)


def _route_for_depot(environment: DMDHHRP_Environment, depot_id: int) -> Optional[Route]:
    for route in environment.routes:
        if route.depot.id == depot_id:
            return route
    return None


@dataclass
class GraphData:
    """动态图输入。"""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_types: torch.Tensor
    depot_indices: torch.Tensor
    patient_indices: torch.Tensor
    pair_index: torch.Tensor
    pair_features: torch.Tensor
    node_objects: List[object]
    node_idx_to_patient_id: Dict[int, int]
    node_idx_to_depot_id: Dict[int, int]
    depot_id_to_node_idx: Dict[int, int]
    patient_id_to_node_idx: Dict[int, int]
    current_time: float
    xy_scale: float
    time_scale: float

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            node_types=self.node_types.to(device),
            depot_indices=self.depot_indices.to(device),
            patient_indices=self.patient_indices.to(device),
            pair_index=self.pair_index.to(device),
            pair_features=self.pair_features.to(device),
            node_objects=self.node_objects,
            node_idx_to_patient_id=self.node_idx_to_patient_id,
            node_idx_to_depot_id=self.node_idx_to_depot_id,
            depot_id_to_node_idx=self.depot_id_to_node_idx,
            patient_id_to_node_idx=self.patient_id_to_node_idx,
            current_time=self.current_time,
            xy_scale=self.xy_scale,
            time_scale=self.time_scale,
        )


def _distance_scale(environment: DMDHHRP_Environment) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for depot in environment.depots:
        xs.append(depot.location.x)
        ys.append(depot.location.y)
    for patient in environment.known_patients:
        xs.append(patient.location.x)
        ys.append(patient.location.y)
    if not xs or not ys:
        return 1.0
    return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def _time_scale(environment: DMDHHRP_Environment) -> float:
    values = [environment.current_time]
    values.extend(depot.max_work_time for depot in environment.depots)
    values.extend(patient.hard_tw[1] for patient in environment.known_patients)
    values.extend(patient.arrival_time for patient in environment.dynamic_patients_pool)
    return max(max(values), 1.0)


def _depot_features(environment: DMDHHRP_Environment, depot: Depot, distance_scale: float, time_scale: float) -> List[float]:
    route = _route_for_depot(environment, depot.id)
    current_load = float(len(route.patients)) if route is not None else 0.0
    remaining_work = depot.max_work_time - max(getattr(route, "current_time", 0.0), getattr(route, "total_time", 0.0)) if route is not None else depot.max_work_time
    frozen_ratio = _normalize(len(route.frozen_nodes), max(len(route.patients), 1)) if route is not None else 0.0
    planned_ratio = _normalize(len(route.planned_nodes), max(len(route.patients), 1)) if route is not None else 0.0
    total_distance = _normalize(getattr(route, "total_distance", 0.0), distance_scale) if route is not None else 0.0
    return [
        _normalize(depot.location.x, distance_scale),
        _normalize(depot.location.y, distance_scale),
        _normalize(environment.current_time, time_scale),
        _normalize(remaining_work, time_scale),
        _normalize(current_load, max(len(environment.known_patients), 1)),
        _normalize(depot.max_work_time, time_scale),
        _normalize(depot.fixed_cost, 1000.0),
        frozen_ratio,
        planned_ratio,
        total_distance,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def _patient_features(
    environment: DMDHHRP_Environment,
    patient: Patient,
    distance_scale: float,
    time_scale: float,
) -> List[float]:
    is_dynamic = 1.0 if patient.is_dynamic else 0.0
    is_scheduled = 1.0 - is_dynamic
    served_flag = 1.0 if patient.is_served else 0.0
    candidate_count = float(len(patient.candidate_centers))
    if not patient.candidate_centers:
        candidate_count = float(len(environment.depots))

    if patient.is_dynamic:
        ready_time = patient.arrival_time
        due_time = patient.arrival_time + patient.max_tolerance_time
        soft_start = patient.arrival_time
        soft_end = patient.arrival_time + patient.ideal_response_time
        waiting_time = max(0.0, environment.current_time - patient.arrival_time)
        urgency = 1.0 - min(waiting_time / max(patient.max_tolerance_time, 1e-9), 1.0)
    else:
        ready_time, due_time = patient.hard_tw
        soft_start, soft_end = patient.soft_tw
        waiting_time = 0.0
        urgency = 0.0

    route_hint = -1.0
    for route in environment.routes:
        for pos, route_patient in enumerate(route.patients):
            if route_patient.id == patient.id:
                route_hint = float(pos)
                break

    return [
        _normalize(patient.location.x, distance_scale),
        _normalize(patient.location.y, distance_scale),
        _normalize(patient.service_time, time_scale),
        _normalize(ready_time, time_scale),
        _normalize(due_time, time_scale),
        _normalize(soft_start, time_scale),
        _normalize(soft_end, time_scale),
        _normalize(waiting_time, time_scale),
        0.0,
        1.0,
        is_dynamic,
        is_scheduled,
        served_flag,
        _normalize(candidate_count, max(len(environment.depots), 1)),
        urgency,
        route_hint,
    ]


def _pair_features(environment: DMDHHRP_Environment, left: object, right: object, distance_scale: float, time_scale: float, is_depot_patient: float, is_patient_patient: float) -> List[float]:
    distance, travel_time = _distance_and_travel_time(left, right, environment.travel_speed)
    return [
        _normalize(travel_time, time_scale),
        _normalize(distance, distance_scale),
        is_depot_patient,
        is_patient_patient,
    ]


def build_graph_from_env(
    environment: DMDHHRP_Environment,
    include_served_patients: bool = True,
    waiting_dynamic: Optional[Sequence[DynamicPatient]] = None,
) -> GraphData:
    """将当前环境转成带节点/边特征的动态图。"""

    distance_scale = _distance_scale(environment)
    time_scale = _time_scale(environment)

    node_objects: List[object] = []
    node_features: List[List[float]] = []
    node_types: List[int] = []
    depot_indices: List[int] = []
    patient_indices: List[int] = []
    node_idx_to_patient_id: Dict[int, int] = {}
    node_idx_to_depot_id: Dict[int, int] = {}
    depot_id_to_node_idx: Dict[int, int] = {}
    patient_id_to_node_idx: Dict[int, int] = {}

    # Depot nodes.
    for depot in environment.depots:
        node_idx = len(node_objects)
        node_objects.append(depot)
        depot_id_to_node_idx[depot.id] = node_idx
        node_idx_to_depot_id[node_idx] = depot.id
        depot_indices.append(node_idx)
        node_types.append(0)
        node_features.append(_depot_features(environment, depot, distance_scale, time_scale))

    # Patient nodes from the environment's known set.
    patient_pool: List[Patient] = list(environment.known_patients)
    if waiting_dynamic is not None:
        seen_ids = {patient.id for patient in patient_pool}
        for patient in waiting_dynamic:
            if patient.id not in seen_ids:
                patient_pool.append(patient)
                seen_ids.add(patient.id)

    for patient in patient_pool:
        if patient.is_served and not include_served_patients:
            continue
        node_idx = len(node_objects)
        node_objects.append(patient)
        patient_id_to_node_idx[patient.id] = node_idx
        node_idx_to_patient_id[node_idx] = patient.id
        patient_indices.append(node_idx)
        node_types.append(1)
        node_features.append(_patient_features(environment, patient, distance_scale, time_scale))

    edge_index: List[List[int]] = []
    edge_features: List[List[float]] = []

    # Depot <-> patient edges.
    for depot in environment.depots:
        depot_idx = depot_id_to_node_idx[depot.id]
        for patient in patient_pool:
            if patient.is_served and not include_served_patients:
                continue
            if patient.candidate_centers and depot.id not in patient.candidate_centers:
                continue
            patient_idx = patient_id_to_node_idx[patient.id]
            left = node_objects[depot_idx]
            right = node_objects[patient_idx]
            feat = _pair_features(environment, left, right, distance_scale, time_scale, 1.0, 0.0)
            edge_index.append([depot_idx, patient_idx])
            edge_features.append(feat)
            edge_index.append([patient_idx, depot_idx])
            edge_features.append(feat)

    # Patient <-> patient edges.
    patients_for_graph = [p for p in patient_pool if include_served_patients or not p.is_served]
    for i, patient_i in enumerate(patients_for_graph):
        idx_i = patient_id_to_node_idx[patient_i.id]
        for j, patient_j in enumerate(patients_for_graph):
            if i == j:
                continue
            idx_j = patient_id_to_node_idx[patient_j.id]
            left = node_objects[idx_i]
            right = node_objects[idx_j]
            feat = _pair_features(environment, left, right, distance_scale, time_scale, 0.0, 1.0)
            edge_index.append([idx_i, idx_j])
            edge_features.append(feat)

    if not edge_index:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_feature_tensor = torch.zeros((0, 4), dtype=torch.float32)
    else:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_feature_tensor = torch.tensor(edge_features, dtype=torch.float32)

    node_feature_tensor = torch.tensor(node_features, dtype=torch.float32) if node_features else torch.zeros((0, 16), dtype=torch.float32)
    node_type_tensor = torch.tensor(node_types, dtype=torch.long) if node_types else torch.zeros((0,), dtype=torch.long)

    # Pair features for patient-depot matching.
    pair_index: List[List[int]] = []
    pair_features: List[List[float]] = []
    for patient in patients_for_graph:
        patient_idx = patient_id_to_node_idx[patient.id]
        for depot in environment.depots:
            depot_idx = depot_id_to_node_idx[depot.id]
            if patient.candidate_centers and depot.id not in patient.candidate_centers:
                continue
            feat = _pair_features(environment, node_objects[patient_idx], node_objects[depot_idx], distance_scale, time_scale, 1.0, 0.0)
            pair_index.append([patient_idx, depot_idx])
            pair_features.append(feat)

    pair_index_tensor = torch.tensor(pair_index, dtype=torch.long) if pair_index else torch.zeros((0, 2), dtype=torch.long)
    pair_feature_tensor = torch.tensor(pair_features, dtype=torch.float32) if pair_features else torch.zeros((0, 4), dtype=torch.float32)

    return GraphData(
        node_features=node_feature_tensor,
        edge_index=edge_index_tensor,
        edge_features=edge_feature_tensor,
        node_types=node_type_tensor,
        depot_indices=torch.tensor(depot_indices, dtype=torch.long),
        patient_indices=torch.tensor(patient_indices, dtype=torch.long),
        pair_index=pair_index_tensor,
        pair_features=pair_feature_tensor,
        node_objects=node_objects,
        node_idx_to_patient_id=node_idx_to_patient_id,
        node_idx_to_depot_id=node_idx_to_depot_id,
        depot_id_to_node_idx=depot_id_to_node_idx,
        patient_id_to_node_idx=patient_id_to_node_idx,
        current_time=environment.current_time,
        xy_scale=distance_scale,
        time_scale=time_scale,
    )


__all__ = ["GraphData", "build_graph_from_env"]