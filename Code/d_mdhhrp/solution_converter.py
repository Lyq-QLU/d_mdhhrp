"""路径格式与 Solution 之间的转换工具。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - package/script dual use
    from .models import DMDHHRP_Environment, Depot, DynamicPatient, Route, ScheduledPatient, Solution
except ImportError:  # pragma: no cover
    from models import DMDHHRP_Environment, Depot, DynamicPatient, Route, ScheduledPatient, Solution


PathType = List[Tuple[int, int]]


def _patient_map(patients: Sequence) -> Dict[int, object]:
    return {patient.id: patient for patient in patients}


def solution_to_path(solution: Solution) -> PathType:
    """将 Solution 按中心/访问顺序转换为 [(center_id, patient_id), ...]。"""
    path: PathType = []
    for center_id, route in solution.routes.items():
        for patient in route.patients:
            path.append((center_id, patient.id))
    return path


def solution_to_dict(solution: Solution) -> Dict[str, object]:
    """将 Solution 转为便于日志记录的字典。"""
    routes = []
    for center_id, route in solution.routes.items():
        routes.append(
            {
                "center_id": center_id,
                "patients": [patient.id for patient in route.patients],
                "service_start_times": list(route.service_start_times),
                "total_distance": route.total_distance,
                "total_time": route.total_time,
                "total_satisfaction": route.total_satisfaction,
                "num_violations": route.num_violations,
            }
        )

    return {
        "routes": routes,
        "unassigned_patients": sorted(solution.unassigned_patients),
        "accepted_dynamic": sorted(solution.accepted_dynamic),
        "rejected_dynamic": sorted(solution.rejected_dynamic),
        "obj1": solution.obj1,
        "obj2": solution.obj2,
        "total_cost": solution.total_cost,
        "total_satisfaction": solution.total_satisfaction,
    }


def build_solution_from_path(
    path: PathType,
    centers: Sequence[Depot],
    patients: Sequence,
    travel_speed: float = 1.0,
    reserve_ratio: float = 0.2,
    R_threshold: float = 15.0,
) -> Solution:
    """从路径构建 Solution，并重新评估其指标。"""
    patient_map = _patient_map(patients)
    route_map = {depot.id: Route(depot) for depot in centers}

    for center_id, patient_id in path:
        route = route_map.get(center_id)
        patient = patient_map.get(patient_id)
        if route is None or patient is None:
            continue
        route.patients.append(patient)

    scheduled_patients = [patient for patient in patients if not getattr(patient, "is_dynamic", False)]
    dynamic_patients = [patient for patient in patients if getattr(patient, "is_dynamic", False)]
    env = DMDHHRP_Environment(
        depots=list(centers),
        scheduled_patients=scheduled_patients,
        dynamic_patients=dynamic_patients,
        R_threshold=R_threshold,
        travel_speed=travel_speed,
        reserve_ratio=reserve_ratio,
    )

    env.routes = [route for route in route_map.values() if route.patients]
    for route in env.routes:
        feasible, start_times, _ = env._sequence_feasible_from_time(route, route.patients)
        route.service_start_times = list(start_times) if feasible else []
        route.sync_planning_state()

    obj1, obj2 = env.evaluate_objectives()
    unassigned = {patient.id for patient in patients if patient.id not in {pid for _, pid in path}}
    solution = Solution(
        routes=route_map,
        unassigned_patients=unassigned,
        accepted_dynamic={patient.id for patient in dynamic_patients if patient.is_served},
        rejected_dynamic={patient.id for patient in dynamic_patients if not patient.is_served},
        obj1=obj1,
        obj2=obj2,
    )
    return solution


def validate_solution_from_path(path: PathType, solution: Solution) -> bool:
    """检查路径与 Solution 的患者集合是否一致。"""
    path_patients = {patient_id for _, patient_id in path}
    solution_patients = {patient.id for route in solution.routes.values() for patient in route.patients}
    return path_patients == solution_patients
