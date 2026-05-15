"""ALNS 破坏 / 修复算子。"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

try:  # pragma: no cover - package/script dual use
    from .models import DMDHHRP_Environment, Patient, Route
except ImportError:  # pragma: no cover
    from models import DMDHHRP_Environment, Patient, Route


def _all_assigned_patients(environment: DMDHHRP_Environment) -> List[Patient]:
    patients: List[Patient] = []
    for route in environment.routes:
        patients.extend(route.patients)
    return patients


def _find_route(environment: DMDHHRP_Environment, patient_id: int) -> Optional[Route]:
    for route in environment.routes:
        for patient in route.patients:
            if patient.id == patient_id:
                return route
    return None


def _remove_patient(environment: DMDHHRP_Environment, patient_id: int) -> Optional[Patient]:
    route = _find_route(environment, patient_id)
    if route is None:
        return None
    removed = route.remove_patient(patient_id)
    if removed is None:
        return None
    removed.is_served = False
    environment.accepted_dynamic.discard(patient_id)
    route.service_start_times = []
    route.sync_planning_state()
    return removed


def _refresh(environment: DMDHHRP_Environment) -> None:
    environment.evaluate_objectives()


class DestroyOperator:
    def __init__(self, name: str):
        self.name = name
        self.success_count = 0
        self.total_count = 0

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        raise NotImplementedError

    def get_success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.5


class RepairOperator:
    def __init__(self, name: str):
        self.name = name
        self.success_count = 0
        self.total_count = 0

    def apply(self, environment: DMDHHRP_Environment, removed_patients: List[int]) -> None:
        raise NotImplementedError

    def get_success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.5


class RandomRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("RandomRemoval")

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        assigned = [patient.id for patient in _all_assigned_patients(environment)]
        if not assigned:
            return []
        num_to_remove = max(1, int(len(assigned) * destroy_ratio))
        removed = random.sample(assigned, min(num_to_remove, len(assigned)))
        for pid in removed:
            _remove_patient(environment, pid)
        _refresh(environment)
        return removed


class ShawRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("ShawRemoval")

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        assigned = [patient.id for patient in _all_assigned_patients(environment)]
        if not assigned:
            return []
        num_to_remove = max(1, int(len(assigned) * destroy_ratio))
        seed = random.choice(assigned)
        removed = [seed]
        relatedness = []
        seed_patient = environment._patient_index[seed]
        seed_route = _find_route(environment, seed)
        for pid in assigned:
            if pid == seed:
                continue
            patient = environment._patient_index[pid]
            distance = seed_patient.location.distance_to(patient.location)
            time_diff = abs(seed_patient.hard_tw[0] - patient.hard_tw[0])
            same_route = 1.0 if _find_route(environment, pid) is seed_route else 0.0
            score = 0.5 * distance + 0.3 * time_diff - 2.0 * same_route
            relatedness.append((pid, score))
        relatedness.sort(key=lambda item: item[1])
        for pid, _ in relatedness[: max(0, num_to_remove - 1)]:
            removed.append(pid)
        for pid in removed:
            _remove_patient(environment, pid)
        _refresh(environment)
        return removed


class WorstRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("WorstRemoval")

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        assigned = [patient.id for patient in _all_assigned_patients(environment)]
        if not assigned:
            return []
        num_to_remove = max(1, int(len(assigned) * destroy_ratio))
        savings = []
        for pid in assigned:
            route = _find_route(environment, pid)
            if route is None:
                continue
            try:
                pos = next(idx for idx, p in enumerate(route.patients) if p.id == pid)
            except StopIteration:
                continue
            patient = environment._patient_index[pid]
            prev_loc = route.depot.location if pos == 0 else route.patients[pos - 1].location
            next_loc = route.depot.location if pos == len(route.patients) - 1 else route.patients[pos + 1].location
            current_cost = prev_loc.distance_to(patient.location) + patient.location.distance_to(next_loc)
            new_cost = prev_loc.distance_to(next_loc)
            savings.append((pid, current_cost - new_cost))
        savings.sort(key=lambda item: item[1], reverse=True)
        removed = [pid for pid, _ in savings[:num_to_remove]]
        for pid in removed:
            _remove_patient(environment, pid)
        _refresh(environment)
        return removed


class WorstSatisfactionRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("WorstSatisfactionRemoval")

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        assigned = [patient.id for patient in _all_assigned_patients(environment)]
        if not assigned:
            return []
        num_to_remove = max(1, int(len(assigned) * destroy_ratio))
        scores = []
        for route in environment.routes:
            feasible, start_times, _ = environment._sequence_feasible_from_time(route, route.patients)
            if not feasible:
                continue
            for patient, start_time in zip(route.patients, start_times):
                sat = patient.calculate_satisfaction(start_time)
                scores.append((patient.id, sat - environment.calculate_spatial_penalty(route.depot, patient)))
        scores.sort(key=lambda item: item[1])
        removed = [pid for pid, _ in scores[:num_to_remove]]
        for pid in removed:
            _remove_patient(environment, pid)
        _refresh(environment)
        return removed


class TimeWindowViolationRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("TimeWindowViolationRemoval")

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        assigned = [patient.id for patient in _all_assigned_patients(environment)]
        if not assigned:
            return []
        num_to_remove = max(1, int(len(assigned) * destroy_ratio))
        violations = []
        for route in environment.routes:
            feasible, start_times, _ = environment._sequence_feasible_from_time(route, route.patients)
            if not feasible:
                for patient in route.patients:
                    violations.append((patient.id, 10.0))
                continue
            for patient, start_time in zip(route.patients, start_times):
                if patient.is_dynamic:
                    penalty = max(0.0, start_time - patient.arrival_time - patient.max_tolerance_time)
                else:
                    penalty = max(0.0, start_time - patient.hard_tw[1])
                violations.append((patient.id, penalty))
        violations.sort(key=lambda item: item[1], reverse=True)
        removed = [pid for pid, _ in violations[:num_to_remove]]
        for pid in removed:
            _remove_patient(environment, pid)
        _refresh(environment)
        return removed


class HighImpactCenterRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("HighImpactCenterRemoval")

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        active_routes = [route for route in environment.routes if route.patients]
        if not active_routes:
            return []
        target = max(active_routes, key=lambda route: len(route.patients))
        num_to_remove = max(1, int(sum(len(route.patients) for route in active_routes) * destroy_ratio))
        candidates = [patient.id for patient in target.patients]
        removed = random.sample(candidates, min(num_to_remove, len(candidates)))
        for pid in removed:
            _remove_patient(environment, pid)
        _refresh(environment)
        return removed


class CenterCloseRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("CenterCloseRemoval")

    def apply(self, environment: DMDHHRP_Environment, destroy_ratio: float = 0.3) -> List[int]:
        active_routes = [route for route in environment.routes if route.patients]
        if len(active_routes) <= 1:
            return RandomRemoval().apply(environment, destroy_ratio=max(0.1, destroy_ratio * 0.5))
        target = min(active_routes, key=lambda route: len(route.patients))
        removed = [patient.id for patient in list(target.patients)]
        for pid in removed:
            _remove_patient(environment, pid)
        _refresh(environment)
        return removed


class GreedyRepair(RepairOperator):
    def __init__(self):
        super().__init__("GreedyRepair")

    def apply(self, environment: DMDHHRP_Environment, removed_patients: List[int]) -> None:
        for pid in list(removed_patients):
            patient = environment._patient_index[pid]
            if environment.greedy_insert(patient):
                removed_patients.remove(pid)
        _refresh(environment)


class Regret2Repair(RepairOperator):
    def __init__(self):
        super().__init__("Regret2Repair")

    def apply(self, environment: DMDHHRP_Environment, removed_patients: List[int]) -> None:
        uninserted = list(removed_patients)
        while uninserted:
            best_patient = None
            best_option = None
            best_regret = -float("inf")
            for pid in uninserted:
                options = []
                patient = environment._patient_index[pid]
                for route_idx, route in enumerate(environment.routes):
                    if patient.candidate_centers and route.depot.id not in patient.candidate_centers:
                        continue
                    for pos in range(len(route.patients) + 1):
                        if pos < len(route.frozen_nodes):
                            continue
                        score = _insertion_score(environment, route_idx, pos, patient)
                        if score is not None:
                            options.append((score, route_idx, pos))
                if not options:
                    continue
                options.sort(key=lambda item: item[0])
                if len(options) == 1:
                    regret = -options[0][0]
                else:
                    regret = options[1][0] - options[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_patient = pid
                    best_option = options[0]
            if best_patient is None or best_option is None:
                break
            _, route_idx, pos = best_option
            patient = environment._patient_index[best_patient]
            if environment.greedy_insert(patient):
                uninserted.remove(best_patient)
                removed_patients.remove(best_patient)
            else:
                uninserted.remove(best_patient)
                removed_patients.remove(best_patient)
        _refresh(environment)


class SatisfactionAwareRepair(RepairOperator):
    def __init__(self):
        super().__init__("SatisfactionAwareRepair")

    def apply(self, environment: DMDHHRP_Environment, removed_patients: List[int]) -> None:
        sorted_patients = sorted(
            removed_patients,
            key=lambda pid: (getattr(environment._patient_index[pid], "arrival_time", 0.0), environment._patient_index[pid].hard_tw[1]),
        )
        for pid in sorted_patients:
            patient = environment._patient_index[pid]
            if environment.greedy_insert(patient):
                removed_patients.remove(pid)
        _refresh(environment)


class TimeWindowFirstRepair(RepairOperator):
    def __init__(self):
        super().__init__("TimeWindowFirstRepair")

    def apply(self, environment: DMDHHRP_Environment, removed_patients: List[int]) -> None:
        sorted_patients = sorted(
            removed_patients,
            key=lambda pid: (
                environment._patient_index[pid].hard_tw[1] if not environment._patient_index[pid].is_dynamic else environment._patient_index[pid].arrival_time + environment._patient_index[pid].max_tolerance_time
            ),
        )
        for pid in sorted_patients:
            patient = environment._patient_index[pid]
            if environment.greedy_insert(patient):
                removed_patients.remove(pid)
        _refresh(environment)


class CenterAwareRegretRepair(Regret2Repair):
    def __init__(self, gamma: float = 30.0):
        super().__init__()
        self.name = "CenterAwareRegretRepair"
        self.gamma = gamma

    def apply(self, environment: DMDHHRP_Environment, removed_patients: List[int]) -> None:
        uninserted = list(removed_patients)
        while uninserted:
            best_patient = None
            best_option = None
            best_regret = -float("inf")
            center_loads = {route.depot.id: len(route.patients) for route in environment.routes}
            avg_load = sum(center_loads.values()) / max(len(center_loads), 1)
            for pid in uninserted:
                options = []
                patient = environment._patient_index[pid]
                for route_idx, route in enumerate(environment.routes):
                    if patient.candidate_centers and route.depot.id not in patient.candidate_centers:
                        continue
                    balance_penalty = self.gamma * max(0.0, center_loads.get(route.depot.id, 0) - avg_load)
                    for pos in range(len(route.patients) + 1):
                        if pos < len(route.frozen_nodes):
                            continue
                        score = _insertion_score(environment, route_idx, pos, patient)
                        if score is not None:
                            options.append((score + balance_penalty, route_idx, pos))
                if not options:
                    continue
                options.sort(key=lambda item: item[0])
                regret = options[1][0] - options[0][0] if len(options) > 1 else -options[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_patient = pid
                    best_option = options[0]
            if best_patient is None or best_option is None:
                break
            patient = environment._patient_index[best_patient]
            if environment.greedy_insert(patient):
                removed_patients.remove(best_patient)
                uninserted.remove(best_patient)
            else:
                removed_patients.remove(best_patient)
                uninserted.remove(best_patient)
        _refresh(environment)


def _insertion_score(environment: DMDHHRP_Environment, route_idx: int, pos: int, patient: Patient) -> Optional[float]:
    if route_idx >= len(environment.routes):
        return None
    route = environment.routes[route_idx]
    if pos < len(route.frozen_nodes):
        return None
    result = environment.evaluate_insertion(route, patient, pos)
    if not result.get("feasible", False):
        return None
    return float(result["delta_cost"] - 100.0 * result["delta_satisfaction"])
