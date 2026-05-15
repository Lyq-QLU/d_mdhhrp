"""GNN 引导 + ALNS 修复 + 滚动重优化的混合求解器。

这个模块把第二阶段方法落到代码层面：
- GNN 先对动态患者插入候选进行打分；
- ALNS 在当前解上做局部扰动与修复；
- 外层仍然通过事件驱动滚动时域调用 `reoptimize(...)`。

目标不是一次性求出全局最优，而是提供一个稳定、可运行、可继续训练/扩展的第二部分方法骨架。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - package/script dual use
    from .gnn_solver import GraphGuidedDynamicSolver
    from .models import DMDHHRP_Environment, DynamicPatient, Route, Solution
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
    from gnn_solver import GraphGuidedDynamicSolver
    from models import DMDHHRP_Environment, DynamicPatient, Route, Solution
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
class _RouteSnapshot:
    depot_id: int
    patient_ids: List[int]
    service_start_times: List[float]
    frozen_nodes: List[int]
    planned_nodes: List[int]
    current_time: float
    current_node: Optional[int]
    total_distance: float
    total_time: float
    total_satisfaction: float
    num_violations: int


@dataclass
class _EnvironmentSnapshot:
    routes: List[_RouteSnapshot]
    accepted_dynamic: List[int]
    rejected_dynamic: List[int]
    current_time: float


class HybridRollingHorizonSolver:
    """Rolling Horizon + GNN guided ALNS 的混合在线求解器。"""

    def __init__(
        self,
        seed: int = 7,
        local_search_iters: int = 3,
        destroy_ratio: float = 0.25,
        accept_threshold: float = -1e9,
        deterministic_policy: bool = True,
    ):
        self.seed = seed
        self.local_search_iters = max(0, int(local_search_iters))
        self.destroy_ratio = max(0.05, min(0.8, float(destroy_ratio)))
        self.deterministic_policy = deterministic_policy

        self.gnn_solver = GraphGuidedDynamicSolver(seed=seed, accept_threshold=accept_threshold)
        self.policy = OperatorSelectorPolicy(seed=seed)

        self.destroy_ops = [
            RandomRemoval(),
            ShawRemoval(),
            WorstRemoval(),
            WorstSatisfactionRemoval(),
            TimeWindowViolationRemoval(),
            HighImpactCenterRemoval(),
            CenterCloseRemoval(),
        ]
        self.repair_ops = [
            GreedyRepair(),
            Regret2Repair(),
            SatisfactionAwareRepair(),
            TimeWindowFirstRepair(),
            CenterAwareRegretRepair(),
            GreedyRepair(),
        ]

        self.last_stage: Dict[str, float] = {}

    def _scalar_score(self, solution: Solution) -> float:
        return float(solution.obj1) - 100.0 * float(solution.obj2)

    def _capture_environment(self, environment: DMDHHRP_Environment) -> _EnvironmentSnapshot:
        route_snaps: List[_RouteSnapshot] = []
        for route in environment.routes:
            route_snaps.append(
                _RouteSnapshot(
                    depot_id=route.depot.id,
                    patient_ids=[patient.id for patient in route.patients],
                    service_start_times=list(route.service_start_times),
                    frozen_nodes=list(route.frozen_nodes),
                    planned_nodes=list(route.planned_nodes),
                    current_time=route.current_time,
                    current_node=route.current_node,
                    total_distance=route.total_distance,
                    total_time=route.total_time,
                    total_satisfaction=route.total_satisfaction,
                    num_violations=route.num_violations,
                )
            )

        return _EnvironmentSnapshot(
            routes=route_snaps,
            accepted_dynamic=sorted(environment.accepted_dynamic),
            rejected_dynamic=sorted(environment.rejected_dynamic),
            current_time=environment.current_time,
        )

    def _restore_environment(self, environment: DMDHHRP_Environment, snapshot: _EnvironmentSnapshot) -> None:
        depot_by_id = {depot.id: depot for depot in environment.depots}
        patient_by_id = environment._patient_index
        restored_routes: List[Route] = []

        for route_snap in snapshot.routes:
            depot = depot_by_id.get(route_snap.depot_id)
            if depot is None:
                continue
            route = Route(depot=depot)
            route.patients = [patient_by_id[pid] for pid in route_snap.patient_ids if pid in patient_by_id]
            route.service_start_times = list(route_snap.service_start_times)
            route.frozen_nodes = list(route_snap.frozen_nodes)
            route.planned_nodes = list(route_snap.planned_nodes)
            route.current_time = route_snap.current_time
            route.current_node = route_snap.current_node
            route.total_distance = route_snap.total_distance
            route.total_time = route_snap.total_time
            route.total_satisfaction = route_snap.total_satisfaction
            route.num_violations = route_snap.num_violations
            restored_routes.append(route)

        environment.routes = restored_routes
        environment.accepted_dynamic = set(snapshot.accepted_dynamic)
        environment.rejected_dynamic = set(snapshot.rejected_dynamic)
        environment.current_time = snapshot.current_time
        environment.evaluate_objectives()

    def _select_operator_pair(
        self,
        environment: DMDHHRP_Environment,
        waiting_dynamic_count: int,
        search_history: Optional[Dict] = None,
    ) -> Tuple[int, int]:
        current_solution = environment._export_solution()
        state = build_state_vector(
            current_solution,
            search_history=search_history,
            waiting_dynamic_count=waiting_dynamic_count,
            life_circle_coverage=0.0,
            rejection_ratio=len(environment.rejected_dynamic) / max(len(environment.known_patients), 1),
        )
        action, _, _ = self.policy.select_action(state, deterministic=self.deterministic_policy)
        destroy_idx, repair_idx = self.policy.decode_action(action)
        destroy_idx %= len(self.destroy_ops)
        repair_idx %= len(self.repair_ops)
        return destroy_idx, repair_idx

    def _apply_local_search(
        self,
        environment: DMDHHRP_Environment,
        waiting_dynamic: Sequence[DynamicPatient],
        current_time: float,
    ) -> Solution:
        best_solution = environment._export_solution()
        best_solution.obj1, best_solution.obj2 = environment.evaluate_objectives()
        best_score = self._scalar_score(best_solution)

        search_history = {
            "iteration": 0,
            "max_iterations": self.local_search_iters,
            "temperature": 1.0,
            "initial_temperature": 1.0,
            "acceptance_rate": 0.5,
            "last_destroy_success": 0.5,
            "last_repair_success": 0.5,
            "diversity_score": 0.5,
            "best_cost": best_solution.obj1,
            "best_satisfaction": best_solution.obj2,
        }

        for iteration in range(self.local_search_iters):
            snapshot = self._capture_environment(environment)
            destroy_idx, repair_idx = self._select_operator_pair(
                environment,
                waiting_dynamic_count=len(waiting_dynamic),
                search_history=search_history,
            )

            removed_patients = self.destroy_ops[destroy_idx].apply(environment, destroy_ratio=self.destroy_ratio)
            if removed_patients:
                self.repair_ops[repair_idx].apply(environment, removed_patients)

            candidate_obj1, candidate_obj2 = environment.evaluate_objectives()
            candidate_solution = environment._export_solution()
            candidate_solution.obj1 = candidate_obj1
            candidate_solution.obj2 = candidate_obj2
            candidate_score = self._scalar_score(candidate_solution)

            improved = (
                candidate_score + 1e-9 < best_score
                or dominates(candidate_solution, best_solution)
                or (
                    candidate_solution.total_satisfaction > best_solution.total_satisfaction
                    and candidate_solution.total_cost <= best_solution.total_cost * 1.02
                )
            )

            if improved:
                best_solution = candidate_solution
                best_score = candidate_score
                search_history["best_cost"] = best_solution.obj1
                search_history["best_satisfaction"] = best_solution.obj2
                search_history["acceptance_rate"] = min(1.0, search_history.get("acceptance_rate", 0.5) + 0.1)
                continue

            self._restore_environment(environment, snapshot)
            search_history["acceptance_rate"] = max(0.0, search_history.get("acceptance_rate", 0.5) - 0.05)
            search_history["iteration"] = iteration + 1

        environment.evaluate_objectives()
        return best_solution

    def reoptimize(
        self,
        current_solution: Solution,
        waiting_dynamic: Sequence[DynamicPatient],
        current_time: float,
        environment: DMDHHRP_Environment,
    ) -> Solution:
        """在线重调度：先 GNN 插入，再 ALNS 做局部优化。"""
        del current_solution  # 保持接口兼容，真正状态以 environment 为准。

        gnn_solution = self.gnn_solver.reoptimize(
            current_solution=environment._export_solution(),
            waiting_dynamic=waiting_dynamic,
            current_time=current_time,
            environment=environment,
        )

        refined_solution = self._apply_local_search(environment, waiting_dynamic, current_time)
        if refined_solution.total_cost <= 0.0 and refined_solution.total_satisfaction <= 0.0:
            refined_solution = gnn_solution

        refined_solution.obj1, refined_solution.obj2 = environment.evaluate_objectives()
        self.last_stage = {
            "current_time": float(current_time),
            "gnn_obj1": float(gnn_solution.obj1),
            "gnn_obj2": float(gnn_solution.obj2),
            "final_obj1": float(refined_solution.obj1),
            "final_obj2": float(refined_solution.obj2),
            "waiting_dynamic": float(len(waiting_dynamic)),
        }
        return refined_solution


__all__ = ["HybridRollingHorizonSolver"]