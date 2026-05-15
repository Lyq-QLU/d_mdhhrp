from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple


class PatientType(str, Enum):
    SCHEDULED = "scheduled"
    DYNAMIC = "dynamic"


@dataclass
class Location:
    x: float
    y: float

    def distance_to(self, other: "Location") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Depot:
    """服务中心 (多中心) - 对应集合 C"""

    id: int
    location: Location
    fixed_cost: float
    max_work_time: float  # H_k: 中心/车辆最大工作时间


@dataclass
class Patient:
    """患者基类 - 对应集合 P"""

    id: int
    location: Location
    service_time: float  # s_i: 服务时长
    is_dynamic: bool = False
    is_served: bool = False  # z_i: 是否被服务决策变量
    hard_tw: Tuple[float, float] = (0.0, float("inf"))
    soft_tw: Tuple[float, float] = (0.0, float("inf"))
    arrival_time: float = 0.0  # 动态患者到达系统的时间 q_i / a_i
    ideal_response_time: float = 15.0
    max_tolerance_time: float = 60.0
    circle_id: int = -1
    candidate_centers: List[int] = field(default_factory=list)

    @property
    def pid(self) -> int:
        return self.id

    @property
    def x(self) -> float:
        return self.location.x

    @property
    def y(self) -> float:
        return self.location.y

    @property
    def release_time(self) -> float:
        return self.arrival_time

    @release_time.setter
    def release_time(self, value: float) -> None:
        self.arrival_time = value

    @property
    def ideal_response(self) -> float:
        return self.ideal_response_time

    @ideal_response.setter
    def ideal_response(self, value: float) -> None:
        self.ideal_response_time = value

    @property
    def max_response(self) -> float:
        return self.max_tolerance_time

    @max_response.setter
    def max_response(self, value: float) -> None:
        self.max_tolerance_time = value

    @property
    def ptype(self) -> PatientType:
        return PatientType.DYNAMIC if self.is_dynamic else PatientType.SCHEDULED

    def calculate_satisfaction(self, service_start_time: float) -> float:
        return 0.0


@dataclass
class ScheduledPatient(Patient):
    """预约患者 - 对应集合 P_0"""

    is_dynamic: bool = False

    def calculate_satisfaction(self, service_start_time: float) -> float:
        """基于硬窗/软窗的梯形满意度函数。"""
        a_i, b_i = self.hard_tw
        l_i, r_i = self.soft_tw

        if l_i <= service_start_time <= r_i:
            return 10.0

        if service_start_time < l_i:
            if service_start_time < a_i:
                return 0.0
            denom = max(l_i - a_i, 1e-9)
            return max(0.0, 10.0 * (service_start_time - a_i) / denom)

        if service_start_time > r_i:
            if service_start_time > b_i:
                return 0.0
            denom = max(b_i - r_i, 1e-9)
            return max(0.0, 10.0 * (b_i - service_start_time) / denom)

        return 0.0


@dataclass
class DynamicPatient(Patient):
    """动态患者 - 对应集合 P_d"""

    is_dynamic: bool = True

    def calculate_satisfaction(self, service_start_time: float) -> float:
        """基于响应时间的梯形满意度函数。"""
        response_time = service_start_time - self.arrival_time
        if response_time < 0:
            return 0.0

        if response_time <= self.ideal_response_time:
            return 10.0

        if response_time <= self.max_tolerance_time:
            denom = max(self.max_tolerance_time - self.ideal_response_time, 1e-9)
            return max(0.0, 10.0 * (self.max_tolerance_time - response_time) / denom)

        return 0.0


@dataclass
class Route:
    """表示一条中心路径。"""

    depot: Depot
    patients: List[Patient] = field(default_factory=list)
    service_start_times: List[float] = field(default_factory=list)
    frozen_nodes: List[int] = field(default_factory=list)
    planned_nodes: List[int] = field(default_factory=list)
    current_time: float = 0.0
    current_node: Optional[int] = None
    total_distance: float = 0.0
    total_time: float = 0.0
    total_satisfaction: float = 0.0
    num_violations: int = 0

    @property
    def center_id(self) -> int:
        return self.depot.id

    def sync_planning_state(self) -> None:
        """保持冻结前缀与计划后缀与当前路径一致。"""
        frozen_set = set(self.frozen_nodes)
        prefix: List[int] = []
        suffix: List[int] = []
        for patient in self.patients:
            (prefix if patient.id in frozen_set else suffix).append(patient.id)
        self.frozen_nodes = prefix
        self.planned_nodes = suffix

    def add_patient(self, patient: Patient, position: Optional[int] = None) -> None:
        if position is None:
            self.patients.append(patient)
        else:
            self.patients.insert(position, patient)

    def remove_patient(self, patient_id: int) -> Optional[Patient]:
        for idx, patient in enumerate(self.patients):
            if patient.id == patient_id:
                return self.patients.pop(idx)
        return None

    def copy(self) -> "Route":
        return Route(
            depot=self.depot,
            patients=list(self.patients),
            service_start_times=list(self.service_start_times),
            frozen_nodes=list(self.frozen_nodes),
            planned_nodes=list(self.planned_nodes),
            current_time=self.current_time,
            current_node=self.current_node,
            total_distance=self.total_distance,
            total_time=self.total_time,
            total_satisfaction=self.total_satisfaction,
            num_violations=self.num_violations,
        )


@dataclass
class Solution:
    routes: Dict[int, Route]
    unassigned_patients: Set[int] = field(default_factory=set)
    accepted_dynamic: Set[int] = field(default_factory=set)
    rejected_dynamic: Set[int] = field(default_factory=set)
    obj1: float = 0.0
    obj2: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.obj1

    @total_cost.setter
    def total_cost(self, value: float) -> None:
        self.obj1 = value

    @property
    def total_satisfaction(self) -> float:
        return self.obj2

    @total_satisfaction.setter
    def total_satisfaction(self, value: float) -> None:
        self.obj2 = value

    def copy(self) -> "Solution":
        return Solution(
            routes={center_id: route.copy() for center_id, route in self.routes.items()},
            unassigned_patients=set(self.unassigned_patients),
            accepted_dynamic=set(self.accepted_dynamic),
            rejected_dynamic=set(self.rejected_dynamic),
            obj1=self.obj1,
            obj2=self.obj2,
        )


class DMDHHRP_Environment:
    """动态多中心居家医疗路径规划问题环境。"""

    def __init__(
        self,
        depots: List[Depot],
        scheduled_patients: List[ScheduledPatient],
        dynamic_patients: List[DynamicPatient],
        R_threshold: float = 15.0,
        travel_speed: float = 1.0,
        reserve_ratio: float = 0.2,
        scheduled_weight: float = 0.5,
        dynamic_weight: float = 0.3,
        life_circle_weight: float = 0.2,
        dynamic_rejection_penalty: float = 1000.0,
        scheduled_rejection_penalty: float = 1e9,
        perturbation_penalty: float = 5.0,
    ):
        self.depots = depots
        self.scheduled_patients = scheduled_patients
        self.dynamic_patients_pool = sorted(dynamic_patients, key=lambda p: p.arrival_time)

        self.current_time = 0.0
        self.known_patients: List[Patient] = list(scheduled_patients)
        self.routes: List[Route] = []

        self.R_threshold = R_threshold
        self.travel_speed = max(travel_speed, 1e-9)
        self.reserve_ratio = reserve_ratio

        self.scheduled_weight = scheduled_weight
        self.dynamic_weight = dynamic_weight
        self.life_circle_weight = life_circle_weight
        self.dynamic_rejection_penalty = dynamic_rejection_penalty
        self.scheduled_rejection_penalty = scheduled_rejection_penalty
        self.perturbation_penalty = perturbation_penalty

        self.accepted_dynamic: Set[int] = set()
        self.rejected_dynamic: Set[int] = set()

        self._known_patient_ids: Set[int] = {p.id for p in self.known_patients}
        self._patient_index: Dict[int, Patient] = {p.id: p for p in self.known_patients}
        for p in self.dynamic_patients_pool:
            self._patient_index[p.id] = p

    def _travel_time(self, from_loc: Location, to_loc: Location) -> float:
        return from_loc.distance_to(to_loc) / self.travel_speed

    def _route_sequence(
        self,
        route: Route,
        additional_patient: Optional[Patient] = None,
        insert_pos: int = -1,
    ) -> List[Patient]:
        patients = list(route.patients)
        if additional_patient is not None:
            if insert_pos < 0:
                insert_pos = len(patients)
            insert_pos = max(insert_pos, len(route.frozen_nodes))
            patients = patients[:insert_pos] + [additional_patient] + patients[insert_pos:]
        return patients

    def _sequence_feasible_from_time(
        self,
        route: Route,
        sequence: Sequence[Patient],
        start_time: float = 0.0,
        start_location: Optional[Location] = None,
    ) -> Tuple[bool, List[float], float]:
        current_time = max(start_time, route.current_time)
        current_loc = start_location if start_location is not None else route.depot.location
        start_times: List[float] = []

        for patient in sequence:
            arrival = current_time + self._travel_time(current_loc, patient.location)
            service_start = arrival

            if patient.is_dynamic:
                service_start = max(service_start, patient.arrival_time)
                if service_start - patient.arrival_time > patient.max_tolerance_time:
                    return False, [], float("inf")
            else:
                service_start = max(service_start, patient.hard_tw[0])
                if service_start > patient.hard_tw[1]:
                    return False, [], float("inf")

            start_times.append(service_start)
            current_time = service_start + patient.service_time
            current_loc = patient.location

        current_time += self._travel_time(current_loc, route.depot.location)
        if current_time > route.depot.max_work_time:
            return False, [], float("inf")

        return True, start_times, current_time

    def _route_objective_components(
        self,
        route: Route,
        start_times: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float, float, float]:
        """返回 (cost, scheduled_dissat, dynamic_dissat, life_circle_penalty)。"""
        if not route.patients:
            return 0.0, 0.0, 0.0, 0.0

        if start_times is None or len(start_times) != len(route.patients):
            feasible, start_times, _ = self._sequence_feasible_from_time(route, route.patients)
            if not feasible:
                return self._calc_route_cost(route) + 1e6, 100.0, 100.0, 100.0

        cost = route.depot.fixed_cost
        sched_dissat = 0.0
        dyn_dissat = 0.0
        lc_penalty = 0.0
        current_loc = route.depot.location

        for patient, s_time in zip(route.patients, start_times):
            cost += self._travel_time(current_loc, patient.location)
            current_loc = patient.location

            sat = patient.calculate_satisfaction(s_time)
            dissat = 10.0 - sat
            lc_penalty += self.calculate_spatial_penalty(route.depot, patient)
            if patient.is_dynamic:
                dyn_dissat += dissat
            else:
                sched_dissat += dissat

        cost += self._travel_time(current_loc, route.depot.location)
        return cost, sched_dissat, dyn_dissat, lc_penalty

    def _build_initial_solution(self) -> Solution:
        """轻量离线初始解：按顺序贪心插入预约患者。"""
        self.routes = []
        for patient in self.scheduled_patients:
            self.greedy_insert(patient)
        return self._export_solution()

    def _export_solution(self) -> Solution:
        accepted_dynamic = {p.id for p in self.known_patients if p.is_dynamic and p.is_served}
        unassigned = {p.id for p in self.known_patients if not p.is_served}
        return Solution(
            routes={route.depot.id: route for route in self.routes},
            unassigned_patients=unassigned,
            accepted_dynamic=accepted_dynamic,
            rejected_dynamic=set(self.rejected_dynamic),
            obj1=0.0,
            obj2=0.0,
        )

    def build_candidate_centers(self, patient: Patient) -> List[int]:
        coverage_pool = list(self.depots)
        seen_ids: Set[int] = set()
        candidate_centers: List[int] = []
        for depot in coverage_pool:
            if depot.id in seen_ids:
                continue
            seen_ids.add(depot.id)
            if self._travel_time(depot.location, patient.location) <= self.R_threshold:
                candidate_centers.append(depot.id)
        if not candidate_centers and coverage_pool:
            nearest = min(coverage_pool, key=lambda depot: self._travel_time(depot.location, patient.location))
            candidate_centers = [nearest.id]
        patient.candidate_centers = candidate_centers
        return candidate_centers

    def freeze_prefixes(self, t: float) -> None:
        """冻结每条路径上已执行/已承诺的前缀。"""
        self.current_time = t
        for route in self.routes:
            if not route.service_start_times or len(route.service_start_times) != len(route.patients):
                feasible, start_times, _ = self._sequence_feasible_from_time(route, route.patients)
                route.service_start_times = list(start_times) if feasible else []

            frozen_count = 0
            current_node: Optional[int] = None
            current_time = t
            for patient, s_time in zip(route.patients, route.service_start_times):
                end_time = s_time + patient.service_time
                if end_time <= t:
                    frozen_count += 1
                    current_node = patient.id
                    current_time = end_time
                elif s_time <= t < end_time:
                    frozen_count += 1
                    current_node = patient.id
                    current_time = t
                    break
                else:
                    break

            route.frozen_nodes = [patient.id for patient in route.patients[:frozen_count]]
            route.planned_nodes = [patient.id for patient in route.patients[frozen_count:]]
            route.current_time = current_time
            route.current_node = current_node

    def update_time(self, t: float):
        """时间推进机制：将到达时间 <= t 的动态患者加入已知集合。"""
        self.current_time = t
        new_arrivals: List[Patient] = []
        for p in self.dynamic_patients_pool:
            if p.arrival_time <= t and p.id not in self._known_patient_ids:
                new_arrivals.append(p)
                self.known_patients.append(p)
                self._known_patient_ids.add(p.id)
        return new_arrivals

    def check_hard_constraints(self, route: Route) -> bool:
        """检查硬约束：预约患者硬时间窗、动态患者响应时限、工作时间约束。"""
        feasible, _, _ = self._sequence_feasible_from_time(route, route.patients)
        return feasible

    def simulate_route(
        self,
        route: Route,
        additional_patient: Optional[Patient] = None,
        insert_pos: int = -1,
    ) -> Tuple[bool, List[float]]:
        """全路径仿真与可行性检查。"""
        sequence = self._route_sequence(route, additional_patient, insert_pos)
        if additional_patient is not None and insert_pos >= 0 and insert_pos < len(route.frozen_nodes):
            return False, []

        feasible, start_times, _ = self._sequence_feasible_from_time(route, sequence)
        return feasible, start_times if feasible else []

    def is_feasible_insertion(self, route: Route, patient: Patient, position: int) -> bool:
        """基于全路径模拟的全局插入可行性检查。"""
        feasible, _ = self.simulate_route(route, patient, position)
        return feasible

    def evaluate_insertion(self, route: Route, patient: Patient, position: int) -> dict:
        """评估插入引起的增量变化。"""
        if position < len(route.frozen_nodes):
            return {"delta_cost": float("inf"), "delta_satisfaction": -float("inf"), "feasible": False}

        feasible, temp_start_times = self.simulate_route(route, patient, position)
        if not feasible:
            return {"delta_cost": float("inf"), "delta_satisfaction": -float("inf"), "feasible": False}

        old_cost, old_sched_dissat, old_dyn_dissat, old_lc = self._route_objective_components(route)
        temp_patients = route.patients[:position] + [patient] + route.patients[position:]
        temp_route = Route(route.depot, temp_patients)
        new_cost, new_sched_dissat, new_dyn_dissat, new_lc = self._route_objective_components(temp_route, temp_start_times)

        old_sat = 30.0 - (old_sched_dissat + old_dyn_dissat + old_lc)
        new_sat = 30.0 - (new_sched_dissat + new_dyn_dissat + new_lc)

        return {"delta_cost": new_cost - old_cost, "delta_satisfaction": new_sat - old_sat, "feasible": True}

    def _calc_route_cost(self, route: Route) -> float:
        if not route.patients:
            return route.depot.fixed_cost

        cost = route.depot.fixed_cost
        curr_loc = route.depot.location
        for p in route.patients:
            cost += self._travel_time(curr_loc, p.location)
            curr_loc = p.location
        cost += self._travel_time(curr_loc, route.depot.location)
        return cost

    def _calc_route_satisfaction(self, route: Route) -> float:
        if not route.patients:
            return 0.0

        feasible, start_times, _ = self._sequence_feasible_from_time(route, route.patients)
        if not feasible:
            return 0.0

        sat = 0.0
        for p, s_time in zip(route.patients, start_times):
            base_sat = p.calculate_satisfaction(s_time)
            sp_pen = self.calculate_spatial_penalty(route.depot, p)
            sat += max(0.0, base_sat - sp_pen)
        return sat

    def calculate_spatial_penalty(self, depot: Depot, patient: Patient) -> float:
        """15 分钟生活圈空间惩罚。"""
        travel_time = self._travel_time(depot.location, patient.location)
        if travel_time > self.R_threshold:
            return (travel_time - self.R_threshold) / max(self.R_threshold, 1e-9)
        return 0.0

    def evaluate_objectives(self) -> Tuple[float, float]:
        """计算双目标函数。"""
        total_distance = 0.0
        total_fixed_cost = 0.0
        total_sched_dissat = 0.0
        total_dyn_dissat = 0.0
        total_lc_pen = 0.0

        for p in self._patient_index.values():
            p.is_served = False

        accepted_dynamic_count = 0
        scheduled_count = len(self.scheduled_patients)

        for route in self.routes:
            if not route.patients:
                total_fixed_cost += route.depot.fixed_cost
                route.total_distance = 0.0
                route.total_time = 0.0
                route.total_satisfaction = 0.0
                route.num_violations = 0
                continue

            total_fixed_cost += route.depot.fixed_cost
            feasible, start_times, _ = self._sequence_feasible_from_time(route, route.patients)
            if not feasible:
                route.total_distance = self._calc_route_cost(route)
                route.total_time = route.total_distance
                route.total_satisfaction = 0.0
                route.num_violations = 1
                total_distance += route.total_distance + 1e6
                total_sched_dissat += 100.0
                total_dyn_dissat += 100.0
                total_lc_pen += 100.0
                continue

            current_loc = route.depot.location
            route.total_distance = 0.0
            route.total_satisfaction = 0.0
            route.num_violations = 0
            for p, start_time in zip(route.patients, start_times):
                total_distance += self._travel_time(current_loc, p.location)
                route.total_distance += self._travel_time(current_loc, p.location)
                current_loc = p.location

                p.is_served = True
                sat = p.calculate_satisfaction(start_time)
                dissat = 10.0 - sat
                route.total_satisfaction += sat
                total_lc_pen += self.calculate_spatial_penalty(route.depot, p)
                if p.is_dynamic:
                    accepted_dynamic_count += 1
                    total_dyn_dissat += dissat
                else:
                    total_sched_dissat += dissat

            total_distance += self._travel_time(current_loc, route.depot.location)
            route.total_distance += self._travel_time(current_loc, route.depot.location)
            route.total_time = route.total_distance

        unserved_penalty_cost = 0.0
        for p in self.known_patients:
            if not p.is_served:
                if p.is_dynamic:
                    unserved_penalty_cost += self.dynamic_rejection_penalty
                    self.rejected_dynamic.add(p.id)
                else:
                    unserved_penalty_cost += self.scheduled_rejection_penalty

        perturbation_cost = 0.0
        f_1 = total_distance + total_fixed_cost + unserved_penalty_cost + self.perturbation_penalty * perturbation_cost

        sched_term = total_sched_dissat / max(scheduled_count, 1)
        dyn_term = total_dyn_dissat / max(accepted_dynamic_count, 1)
        lc_term = total_lc_pen / max(scheduled_count + accepted_dynamic_count, 1)
        f_2 = self.scheduled_weight * sched_term + self.dynamic_weight * dyn_term + self.life_circle_weight * lc_term

        self.accepted_dynamic = {p.id for p in self.known_patients if p.is_dynamic and p.is_served}
        return f_1, f_2

    def greedy_insert(self, patient: Patient) -> bool:
        """贪心插入：遍历所有开启路径与所有可行插入位置。"""
        if not patient.candidate_centers:
            self.build_candidate_centers(patient)

        best_route_idx = -1
        best_insert_pos = -1
        best_obj_val = float("inf")

        weight_cost = 1.0
        weight_sat = 100.0

        for r_idx, route in enumerate(self.routes):
            if patient.candidate_centers and route.depot.id not in patient.candidate_centers:
                continue

            for pos in range(len(route.patients) + 1):
                if pos < len(route.frozen_nodes):
                    continue
                feasible, _ = self.simulate_route(route, patient, pos)
                if feasible:
                    eval_res = self.evaluate_insertion(route, patient, pos)
                    combined_obj = weight_cost * eval_res["delta_cost"] - weight_sat * eval_res["delta_satisfaction"]
                    if combined_obj < best_obj_val:
                        best_obj_val = combined_obj
                        best_route_idx = r_idx
                        best_insert_pos = pos

        if best_route_idx != -1:
            target_route = self.routes[best_route_idx]
            target_route.patients.insert(best_insert_pos, patient)
            feasible, start_times = self.simulate_route(target_route)
            if feasible:
                target_route.service_start_times = start_times
                target_route.sync_planning_state()
                return True
            target_route.patients.pop(best_insert_pos)
            return False

        for depot in self.depots:
            if patient.candidate_centers and depot.id not in patient.candidate_centers:
                continue

            new_route = Route(depot)
            feasible, start_times = self.simulate_route(new_route, patient, 0)
            if feasible:
                new_route.patients.append(patient)
                new_route.service_start_times = start_times
                new_route.sync_planning_state()
                self.routes.append(new_route)
                return True

        return False

    def run_one_day(self, offline_solver=None, online_solver=None) -> Solution:
        """事件驱动运行一天：先离线预约，再按动态到达逐步重优化。"""
        if offline_solver is not None and hasattr(offline_solver, "solve"):
            result = offline_solver.solve(self)
            if isinstance(result, Solution):
                self.routes = list(result.routes.values())
        elif not self.routes:
            self._build_initial_solution()

        events = list(self.dynamic_patients_pool)
        waiting_dynamic: List[DynamicPatient] = []

        for event in events:
            self.update_time(event.arrival_time)
            self.freeze_prefixes(self.current_time)
            waiting_dynamic.append(event)

            still_waiting: List[DynamicPatient] = []
            for p in waiting_dynamic:
                if self.current_time - p.arrival_time > p.max_tolerance_time:
                    self.rejected_dynamic.add(p.id)
                else:
                    still_waiting.append(p)
            waiting_dynamic = still_waiting

            if online_solver is not None and hasattr(online_solver, "reoptimize"):
                maybe_solution = online_solver.reoptimize(
                    current_solution=self._export_solution(),
                    waiting_dynamic=waiting_dynamic,
                    current_time=self.current_time,
                    environment=self,
                )
                if isinstance(maybe_solution, Solution):
                    self.routes = list(maybe_solution.routes.values())
                    served_dynamic_ids = {
                        patient.id
                        for route in self.routes
                        for patient in route.patients
                        if patient.is_dynamic and patient.id in {p.id for p in waiting_dynamic}
                    }
                    waiting_dynamic = [p for p in waiting_dynamic if p.id not in served_dynamic_ids]
            else:
                inserted: List[DynamicPatient] = []
                for p in waiting_dynamic:
                    if self.greedy_insert(p):
                        self.accepted_dynamic.add(p.id)
                        inserted.append(p)
                waiting_dynamic = [p for p in waiting_dynamic if p not in inserted]

        obj1, obj2 = self.evaluate_objectives()
        solution = self._export_solution()
        solution.obj1 = obj1
        solution.obj2 = obj2
        return solution
