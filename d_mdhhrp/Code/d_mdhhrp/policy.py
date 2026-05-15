"""算子选择策略与奖励函数（纯标准库版本）。"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class OperatorSelectorPolicy:
    """轻量算子选择策略。

    说明：
    - 这个版本不依赖 numpy / torch，便于在当前环境直接导入。
    - 接口保持和 DR-GA 的策略类相近，后续如果安装深度学习库，可以在不改调用侧的前提下替换实现。
    """

    def __init__(
        self,
        state_dim: int = 24,
        num_destroy_ops: int = 7,
        num_repair_ops: int = 6,
        hidden_dim: int = 128,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.num_destroy_ops = num_destroy_ops
        self.num_repair_ops = num_repair_ops
        self.num_actions = num_destroy_ops * num_repair_ops
        self.hidden_dim = hidden_dim
        self.rng = random.Random(seed)

        # 纯 Python 线性策略：weights[action][feature]
        self.actor_weights = [
            [self.rng.uniform(-0.05, 0.05) for _ in range(state_dim)]
            for _ in range(self.num_actions)
        ]
        self.actor_bias = [0.0 for _ in range(self.num_actions)]
        self.critic_weights = [self.rng.uniform(-0.05, 0.05) for _ in range(state_dim)]
        self.critic_bias = 0.0

    def train(self):
        return self

    def eval(self):
        return self

    @staticmethod
    def _softmax(logits: Sequence[float]) -> List[float]:
        if not logits:
            return []
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        total = sum(exps) or 1.0
        return [x / total for x in exps]

    def _dot(self, weights: Sequence[float], state: Sequence[float]) -> float:
        return sum(w * float(s) for w, s in zip(weights, state))

    def forward(self, state: Sequence[float]) -> Tuple[List[float], float]:
        state = list(state)
        if len(state) < self.state_dim:
            state = state + [0.0] * (self.state_dim - len(state))
        elif len(state) > self.state_dim:
            state = state[: self.state_dim]

        logits = [self._dot(weights, state) + bias for weights, bias in zip(self.actor_weights, self.actor_bias)]
        action_probs = self._softmax(logits)
        value = self._dot(self.critic_weights, state) + self.critic_bias
        return action_probs, value

    def select_action(self, state: Sequence[float], deterministic: bool = False) -> Tuple[int, float, float]:
        action_probs, value = self.forward(state)
        if not action_probs:
            return 0, 0.0, value

        if deterministic:
            action = max(range(len(action_probs)), key=lambda idx: action_probs[idx])
        else:
            action = self.rng.choices(range(len(action_probs)), weights=action_probs, k=1)[0]

        log_prob = math.log(action_probs[action] + 1e-8)
        return action, log_prob, value

    def decode_action(self, action: int) -> Tuple[int, int]:
        destroy_idx = action // self.num_repair_ops
        repair_idx = action % self.num_repair_ops
        return destroy_idx, repair_idx

    def encode_action(self, destroy_idx: int, repair_idx: int) -> int:
        return destroy_idx * self.num_repair_ops + repair_idx


class ReplayBuffer:
    """经验回放缓冲区。"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self) -> int:
        return len(self.buffer)


def dominates(sol1, sol2) -> bool:
    """判断 sol1 是否支配 sol2（成本最小化，满意度最大化）。"""
    cost_better = sol1.total_cost <= sol2.total_cost
    sat_better = sol1.total_satisfaction >= sol2.total_satisfaction
    strictly_better = sol1.total_cost < sol2.total_cost or sol1.total_satisfaction > sol2.total_satisfaction
    return cost_better and sat_better and strictly_better


def is_pareto_optimal(solution, solution_pool: list) -> bool:
    if len(solution_pool) == 0:
        return True
    for other in solution_pool:
        if dominates(other, solution):
            return False
    return True


def compute_diversity_score(solution, solution_pool: list, jaccard_weight: float = 0.6, objective_weight: float = 0.4) -> float:
    if len(solution_pool) == 0:
        return 1.0

    solution_patients = set()
    for route in solution.routes.values():
        solution_patients.update(patient.id for patient in route.patients)

    diversity_scores = []
    for other in solution_pool:
        other_patients = set()
        for route in other.routes.values():
            other_patients.update(patient.id for patient in route.patients)

        if len(solution_patients) == 0 and len(other_patients) == 0:
            jaccard_dist = 0.0
        else:
            union = len(solution_patients | other_patients)
            intersection = len(solution_patients & other_patients)
            jaccard_dist = 1.0 - (intersection / union if union > 0 else 0.0)

        cost_diff = abs(solution.total_cost - other.total_cost) / max(solution.total_cost, other.total_cost, 1.0)
        sat_diff = abs(solution.total_satisfaction - other.total_satisfaction) / 10.0
        objective_dist = math.sqrt(cost_diff * cost_diff + sat_diff * sat_diff)
        diversity_scores.append(jaccard_weight * jaccard_dist + objective_weight * objective_dist)

    return sum(diversity_scores) / max(len(diversity_scores), 1)


def build_state_vector(
    solution,
    search_history: Optional[Dict] = None,
    waiting_dynamic_count: int = 0,
    life_circle_coverage: float = 0.0,
    rejection_ratio: float = 0.0,
) -> List[float]:
    """把当前搜索状态压成固定长度特征向量。"""
    search_history = search_history or {}

    total_patients = len(getattr(solution, "unassigned_patients", [])) + sum(len(route.patients) for route in solution.routes.values())
    assigned_patients = total_patients - len(getattr(solution, "unassigned_patients", []))
    coverage = assigned_patients / max(total_patients, 1)

    features = [
        float(solution.total_cost) / 3000.0,
        float(solution.total_satisfaction) / 10.0,
        len(getattr(solution, "unassigned_patients", [])) / max(total_patients, 1),
        coverage,
        len(getattr(solution, "accepted_dynamic", [])) / max(total_patients, 1),
        len(getattr(solution, "rejected_dynamic", [])) / max(total_patients, 1),
        float(waiting_dynamic_count) / max(total_patients, 1),
        float(search_history.get("iteration", 0)) / max(search_history.get("max_iterations", 1), 1),
        float(search_history.get("iterations_since_improvement", 0)) / max(search_history.get("max_iterations", 1), 1),
        float(search_history.get("temperature", 1.0)) / max(search_history.get("initial_temperature", 1.0), 1e-9),
        float(search_history.get("acceptance_rate", 0.5)),
        float(search_history.get("last_destroy_success", 0.5)),
        float(search_history.get("last_repair_success", 0.5)),
        float(search_history.get("diversity_score", 0.5)),
        float(life_circle_coverage),
        float(rejection_ratio),
        float(search_history.get("avg_improvement", 0.0)),
        float(search_history.get("best_cost", solution.total_cost)) / 3000.0,
        float(search_history.get("best_satisfaction", solution.total_satisfaction)) / 10.0,
    ]

    while len(features) < 24:
        features.append(0.0)

    return features[:24]


def compute_alns_reward(
    old_solution,
    new_solution,
    is_accepted: bool,
    pareto_solutions: list,
    destroy_op_name: str | None = None,
    repair_op_name: str | None = None,
) -> float:
    """计算 ALNS-DRL 奖励。"""
    if len(getattr(new_solution, "unassigned_patients", [])) > 0:
        return -5.0

    sat_ops = {"satisfactionawarerepair", "timewindowfirstrepair", "centerawareregretrepair"}

    if len(pareto_solutions) > 0:
        min_cost = min(s.total_cost for s in pareto_solutions)
        max_sat = max(s.total_satisfaction for s in pareto_solutions)
        if new_solution.total_cost < min_cost:
            return +5.0
        if new_solution.total_satisfaction > max_sat:
            reward = +6.0
            if repair_op_name and repair_op_name.lower() in sat_ops:
                reward += 1.0
            return reward

    sat_improve = new_solution.total_satisfaction - old_solution.total_satisfaction
    cost_degrade = new_solution.total_cost - old_solution.total_cost
    if sat_improve > 0.4 and cost_degrade <= max(old_solution.total_cost * 0.03, 1e-9):
        return +3.5

    cost_better = new_solution.total_cost <= old_solution.total_cost
    sat_better = new_solution.total_satisfaction >= old_solution.total_satisfaction
    strictly_better = new_solution.total_cost < old_solution.total_cost or new_solution.total_satisfaction > old_solution.total_satisfaction
    if cost_better and sat_better and strictly_better:
        return +3.0

    if is_accepted:
        if len(pareto_solutions) > 0:
            min_cost = min(s.total_cost for s in pareto_solutions)
            max_sat = max(s.total_satisfaction for s in pareto_solutions)
            cost_gap = (new_solution.total_cost - min_cost) / max(min_cost, 1.0)
            sat_gap = (max_sat - new_solution.total_satisfaction) / max(max_sat, 1.0)
            if cost_gap < 0.1 or sat_gap < 0.1:
                return +1.5
        return +1.0

    if len(pareto_solutions) > 0:
        min_cost = min(s.total_cost for s in pareto_solutions)
        max_sat = max(s.total_satisfaction for s in pareto_solutions)
        cost_gap = (new_solution.total_cost - min_cost) / max(min_cost, 1.0)
        sat_gap = (max_sat - new_solution.total_satisfaction) / max(max_sat, 1.0)
        if cost_gap < 0.15 or sat_gap < 0.15:
            return +0.5

    return -0.5
