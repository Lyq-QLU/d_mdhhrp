"""事件驱动的动态调度仿真入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - package/script dual use
    from .models import DMDHHRP_Environment, Solution
except ImportError:  # pragma: no cover
    from models import DMDHHRP_Environment, Solution


@dataclass
class SimulationResult:
    solution: Solution
    history: List[Dict[str, Any]] = field(default_factory=list)


class DynamicSchedulingSimulator:
    """把环境中的动态到达、冻结和重优化流程包成一个可复用接口。"""

    def __init__(self, environment: DMDHHRP_Environment):
        self.environment = environment

    def run(
        self,
        offline_solver: Optional[Any] = None,
        online_solver: Optional[Any] = None,
    ) -> SimulationResult:
        solution = self.environment.run_one_day(offline_solver=offline_solver, online_solver=online_solver)
        return SimulationResult(solution=solution, history=[])

    def advance_to(self, current_time: float) -> List:
        """推进时间并返回新到达的动态患者。"""
        new_arrivals = self.environment.update_time(current_time)
        self.environment.freeze_prefixes(current_time)
        return new_arrivals
