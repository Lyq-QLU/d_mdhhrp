"""基于图神经网络风格消息传递的动态患者在线求解器。

说明：
- 这个实现不依赖 torch / numpy，使用纯 Python 完成可运行的 GNN 风格消息传递。
- 核心思路是：把中心、路径摘要、患者、等待动态患者构造成图，
  用消息传递编码当前状态，再对候选插入动作打分。
- 该模块的定位是第二章方法骨架：GNN 负责打分，环境负责可行性约束。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - package/script dual use
    from .models import DMDHHRP_Environment, DynamicPatient, Patient, Route, Solution
except ImportError:  # pragma: no cover
    from models import DMDHHRP_Environment, DynamicPatient, Patient, Route, Solution


Vector = List[float]


def _zeros(dim: int) -> Vector:
    return [0.0 for _ in range(dim)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    return [_dot(row, vector) for row in matrix]


def _vec_add(*vectors: Sequence[float]) -> Vector:
    if not vectors:
        return []
    dim = len(vectors[0])
    result = [0.0] * dim
    for vec in vectors:
        for idx, value in enumerate(vec):
            result[idx] += float(value)
    return result


def _vec_scale(vector: Sequence[float], scale: float) -> Vector:
    return [float(v) * scale for v in vector]


def _tanh_vector(vector: Sequence[float]) -> Vector:
    return [math.tanh(float(v)) for v in vector]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize(value: float, scale: float) -> float:
    return float(value) / max(scale, 1e-9)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class GraphNode:
    key: str
    kind: str
    features: Vector
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSnapshot:
    """图快照：节点 + 邻接表。"""

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    adjacency: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    current_time: float = 0.0
    xy_scale: float = 1.0
    time_scale: float = 1.0

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.key] = node
        self.adjacency.setdefault(node.key, [])

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        self.adjacency.setdefault(source, []).append((target, relation, weight))


class PurePythonGraphEncoder:
    """纯 Python 消息传递编码器。

    目标不是追求训练级性能，而是提供一个结构完整、可运行、易替换为
    torch 实现的 GNN 骨架。
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, num_layers: int = 2, seed: int = 7):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rng = random.Random(seed)

        self.input_w = self._random_matrix(hidden_dim, feature_dim, scale=0.12)
        self.input_b = [self.rng.uniform(-0.02, 0.02) for _ in range(hidden_dim)]
        self.self_weights = [self._random_matrix(hidden_dim, hidden_dim, scale=0.08) for _ in range(num_layers)]
        self.neigh_weights = [self._random_matrix(hidden_dim, hidden_dim, scale=0.08) for _ in range(num_layers)]
        self.biases = [[self.rng.uniform(-0.01, 0.01) for _ in range(hidden_dim)] for _ in range(num_layers)]

        self.relation_scales: Dict[str, float] = {
            "route_depot": 1.20,
            "route_patient": 1.00,
            "patient_patient": 0.75,
            "depot_patient": 0.95,
            "waiting_route": 1.10,
            "waiting_depot": 0.90,
            "global": 0.50,
        }

    def _random_matrix(self, rows: int, cols: int, scale: float = 0.1) -> List[Vector]:
        return [[self.rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    def encode(self, snapshot: GraphSnapshot) -> Dict[str, Vector]:
        embeddings: Dict[str, Vector] = {}
        for key, node in snapshot.nodes.items():
            features = list(node.features)
            if len(features) < self.feature_dim:
                features.extend([0.0] * (self.feature_dim - len(features)))
            else:
                features = features[: self.feature_dim]
            embeddings[key] = _tanh_vector(_vec_add(_matvec(self.input_w, features), self.input_b))

        for layer_idx in range(self.num_layers):
            next_embeddings: Dict[str, Vector] = {}
            for key, node in snapshot.nodes.items():
                self_term = _matvec(self.self_weights[layer_idx], embeddings[key])

                neighbor_vectors: List[Vector] = []
                neighbor_weights: List[float] = []
                for neighbor_key, relation, weight in snapshot.adjacency.get(key, []):
                    neighbor_emb = embeddings.get(neighbor_key)
                    if neighbor_emb is None:
                        continue
                    scale = self.relation_scales.get(relation, 1.0) * float(weight)
                    neighbor_vectors.append(_vec_scale(neighbor_emb, scale))
                    neighbor_weights.append(abs(scale))

                if neighbor_vectors:
                    agg = _zeros(self.hidden_dim)
                    total = sum(neighbor_weights) or 1.0
                    for vec in neighbor_vectors:
                        for idx, value in enumerate(vec):
                            agg[idx] += value
                    agg = [value / total for value in agg]
                else:
                    agg = _zeros(self.hidden_dim)

                neigh_term = _matvec(self.neigh_weights[layer_idx], agg)
                next_embeddings[key] = _tanh_vector(_vec_add(self_term, neigh_term, self.biases[layer_idx]))

            embeddings = next_embeddings

        return embeddings


@dataclass
class GraphReoptimizationResult:
    solution: Solution
    served_dynamic_ids: List[int] = field(default_factory=list)


class GraphGuidedDynamicSolver:
    """GNN 风格的动态患者在线重调度器。

    接口：
    - `reoptimize(...)`：与 `DMDHHRP_Environment.run_one_day()` 的在线求解接口对接。
    - 返回 `Solution`，并可通过 `served_dynamic_ids` 帮助外层同步 waiting 列表。
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        action_hidden_dim: int = 24,
        seed: int = 7,
        accept_threshold: float = -1e9,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_hidden_dim = action_hidden_dim
        self.accept_threshold = accept_threshold
        self.rng = random.Random(seed)

        self.encoder = PurePythonGraphEncoder(feature_dim=15, hidden_dim=hidden_dim, num_layers=num_layers, seed=seed)

        action_input_dim = hidden_dim * 2 + 10
        self.action_w_patient = self._random_matrix(action_hidden_dim, hidden_dim, scale=0.10)
        self.action_w_route = self._random_matrix(action_hidden_dim, hidden_dim, scale=0.10)
        self.action_w_feat = self._random_matrix(action_hidden_dim, 10, scale=0.10)
        self.action_b = [self.rng.uniform(-0.01, 0.01) for _ in range(action_hidden_dim)]
        self.action_v = [self.rng.uniform(-0.08, 0.08) for _ in range(action_hidden_dim)]
        self.action_v_bias = self.rng.uniform(-0.01, 0.01)

        self.last_snapshot: Optional[GraphSnapshot] = None
        self.last_embeddings: Dict[str, Vector] = {}
        self.last_served_dynamic_ids: List[int] = []

    def _random_matrix(self, rows: int, cols: int, scale: float = 0.1) -> List[Vector]:
        return [[self.rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    def _project_action(self, patient_emb: Sequence[float], route_emb: Sequence[float], features: Sequence[float]) -> float:
        hidden = _vec_add(
            _matvec(self.action_w_patient, patient_emb),
            _matvec(self.action_w_route, route_emb),
            _matvec(self.action_w_feat, features),
            self.action_b,
        )
        hidden = _tanh_vector(hidden)
        return _dot(self.action_v, hidden) + self.action_v_bias

    @staticmethod
    def _softmax(logits: Sequence[float]) -> List[float]:
        if not logits:
            return []
        max_logit = max(logits)
        exps = [math.exp(float(value) - max_logit) for value in logits]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def _action_forward(
        self,
        patient_emb: Sequence[float],
        route_emb: Sequence[float],
        features: Sequence[float],
    ) -> Dict[str, Any]:
        z = _vec_add(
            _matvec(self.action_w_patient, patient_emb),
            _matvec(self.action_w_route, route_emb),
            _matvec(self.action_w_feat, features),
            self.action_b,
        )
        h = _tanh_vector(z)
        score = _dot(self.action_v, h) + self.action_v_bias
        return {"z": z, "h": h, "score": score}

    def _collect_all_patients(self, environment: DMDHHRP_Environment) -> List[Patient]:
        return list(environment.known_patients)

    def _distance_scale(self, environment: DMDHHRP_Environment) -> float:
        xs = [depot.location.x for depot in environment.depots]
        ys = [depot.location.y for depot in environment.depots]
        for patient in environment.known_patients:
            xs.append(patient.location.x)
            ys.append(patient.location.y)
        if not xs or not ys:
            return 1.0
        return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    def _time_scale(self, environment: DMDHHRP_Environment) -> float:
        candidates = [environment.current_time]
        candidates.extend(depot.max_work_time for depot in environment.depots)
        candidates.extend(patient.hard_tw[1] for patient in environment.known_patients)
        candidates.extend(patient.arrival_time for patient in environment.dynamic_patients_pool)
        return max(max(candidates), 1.0)

    def _patient_node_features(self, patient: Patient, snapshot: GraphSnapshot, distance_scale: float, time_scale: float, assigned_route_id: Optional[int]) -> Vector:
        kind_flags = [0.0, 0.0, 1.0, 0.0] if not patient.is_dynamic else [0.0, 0.0, 0.0, 1.0]
        center_count = len(patient.candidate_centers)
        candidate_ratio = _normalize(center_count, max(len(snapshot.nodes), 1))
        time_to_now = snapshot.current_time - patient.arrival_time
        waiting_ratio = 0.0 if not patient.is_dynamic else _clamp(time_to_now / max(patient.max_tolerance_time, 1e-9), 0.0, 1.0)
        remaining_slack = max(patient.max_tolerance_time - max(time_to_now, 0.0), 0.0) if patient.is_dynamic else max(patient.hard_tw[1] - snapshot.current_time, 0.0)
        served_flag = 1.0 if assigned_route_id is not None else 0.0
        route_hint = _normalize(assigned_route_id if assigned_route_id is not None else -1, max(len(snapshot.nodes), 1))
        return [
            *kind_flags,
            _normalize(patient.location.x, distance_scale),
            _normalize(patient.location.y, distance_scale),
            _normalize(patient.service_time, time_scale),
            _normalize(patient.hard_tw[0], time_scale),
            _normalize(patient.hard_tw[1], time_scale),
            _normalize(patient.arrival_time, time_scale),
            waiting_ratio,
            _normalize(remaining_slack, time_scale),
            candidate_ratio,
            served_flag,
            route_hint,
        ]

    def _route_node_features(self, route: Route, snapshot: GraphSnapshot, distance_scale: float, time_scale: float) -> Vector:
        frozen_ratio = _normalize(len(route.frozen_nodes), max(len(route.patients), 1))
        patient_count = _normalize(len(route.patients), max(len(snapshot.nodes), 1))
        distance = _normalize(route.total_distance, distance_scale)
        total_time = _normalize(route.total_time or route.current_time, time_scale)
        satisfaction = _normalize(route.total_satisfaction, 10.0 * max(len(route.patients), 1))
        slack = _normalize(max(route.depot.max_work_time - max(route.total_time, route.current_time), 0.0), time_scale)
        current_node_hint = _normalize(route.current_node if route.current_node is not None else -1, max(len(snapshot.nodes), 1))
        return [
            1.0,
            0.0,
            _normalize(route.depot.location.x, distance_scale),
            _normalize(route.depot.location.y, distance_scale),
            patient_count,
            distance,
            total_time,
            satisfaction,
            frozen_ratio,
            slack,
            current_node_hint,
            _normalize(route.depot.max_work_time, time_scale),
            _normalize(route.depot.fixed_cost, 1000.0),
            _normalize(len(route.service_start_times), max(len(route.patients), 1)),
        ]

    def _depot_node_features(self, depot, snapshot: GraphSnapshot, distance_scale: float, time_scale: float) -> Vector:
        route = next((r for r in snapshot.payload.get("routes", []) if r.depot.id == depot.id), None)
        if route is None:
            return [
                1.0,
                _normalize(depot.location.x, distance_scale),
                _normalize(depot.location.y, distance_scale),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                _normalize(depot.max_work_time, time_scale),
                _normalize(depot.fixed_cost, 1000.0),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        route_count = _normalize(len(route.patients), max(len(snapshot.nodes), 1))
        route_slack = _normalize(max(depot.max_work_time - max(route.total_time, route.current_time), 0.0), time_scale)
        return [
            1.0,
            _normalize(depot.location.x, distance_scale),
            _normalize(depot.location.y, distance_scale),
            route_count,
            _normalize(depot.max_work_time, time_scale),
            _normalize(depot.fixed_cost, 1000.0),
            route_slack,
            _normalize(route.total_distance, distance_scale),
            _normalize(route.total_satisfaction, 10.0 * max(len(route.patients), 1)),
            _normalize(len(route.frozen_nodes), max(len(route.patients), 1)),
            _normalize(route.current_time, time_scale),
            _normalize(route.current_node if route.current_node is not None else -1, max(len(snapshot.nodes), 1)),
            _normalize(len(route.service_start_times), max(len(route.patients), 1)),
            1.0,
        ]

    def _build_snapshot(
        self,
        environment: DMDHHRP_Environment,
        waiting_dynamic: Sequence[DynamicPatient],
        current_time: float,
    ) -> GraphSnapshot:
        distance_scale = self._distance_scale(environment)
        time_scale = self._time_scale(environment)
        snapshot = GraphSnapshot(
            current_time=current_time,
            xy_scale=distance_scale,
            time_scale=time_scale,
            payload={"routes": list(environment.routes), "waiting_dynamic": list(waiting_dynamic)},
        )

        assigned_route_by_patient: Dict[int, int] = {}
        for route in environment.routes:
            for patient in route.patients:
                assigned_route_by_patient[patient.id] = route.depot.id

        # Depot nodes.
        for depot in environment.depots:
            snapshot.add_node(
                GraphNode(
                    key=f"depot:{depot.id}",
                    kind="depot",
                    features=self._depot_node_features(depot, snapshot, distance_scale, time_scale),
                    payload={"depot": depot},
                )
            )

        # Route summary nodes.
        for route in environment.routes:
            route_key = f"route:{route.depot.id}"
            snapshot.add_node(
                GraphNode(
                    key=route_key,
                    kind="route",
                    features=self._route_node_features(route, snapshot, distance_scale, time_scale),
                    payload={"route": route},
                )
            )

        # Patient nodes.
        for patient in self._collect_all_patients(environment):
            patient_key = f"patient:{patient.id}"
            snapshot.add_node(
                GraphNode(
                    key=patient_key,
                    kind="patient",
                    features=self._patient_node_features(
                        patient,
                        snapshot,
                        distance_scale,
                        time_scale,
                        assigned_route_by_patient.get(patient.id),
                    ),
                    payload={"patient": patient},
                )
            )

        # Depot <-> route links.
        for route in environment.routes:
            depot_key = f"depot:{route.depot.id}"
            route_key = f"route:{route.depot.id}"
            snapshot.add_edge(depot_key, route_key, "route_depot", 1.0)
            snapshot.add_edge(route_key, depot_key, "route_depot", 1.0)

        # Route <-> patient links, and patient sequence links.
        for route in environment.routes:
            route_key = f"route:{route.depot.id}"
            for idx, patient in enumerate(route.patients):
                patient_key = f"patient:{patient.id}"
                snapshot.add_edge(route_key, patient_key, "route_patient", 1.0)
                snapshot.add_edge(patient_key, route_key, "route_patient", 1.0)

                if idx > 0:
                    prev_patient = route.patients[idx - 1]
                    prev_key = f"patient:{prev_patient.id}"
                    snapshot.add_edge(prev_key, patient_key, "patient_patient", 1.0)
                    snapshot.add_edge(patient_key, prev_key, "patient_patient", 1.0)

        # Depot <-> patient links for coverage / access relations.
        for patient in self._collect_all_patients(environment):
            patient_key = f"patient:{patient.id}"
            for depot in environment.depots:
                if patient.candidate_centers and depot.id not in patient.candidate_centers:
                    continue
                dist = environment._travel_time(depot.location, patient.location)
                weight = 1.0 / (1.0 + dist)
                depot_key = f"depot:{depot.id}"
                snapshot.add_edge(depot_key, patient_key, "depot_patient", weight)
                snapshot.add_edge(patient_key, depot_key, "depot_patient", weight)

        # Waiting dynamic <-> route / depot links.
        for patient in waiting_dynamic:
            patient_key = f"patient:{patient.id}"
            if patient_key not in snapshot.nodes:
                continue
            for route in environment.routes:
                if patient.candidate_centers and route.depot.id not in patient.candidate_centers:
                    continue
                route_key = f"route:{route.depot.id}"
                dist = environment._travel_time(route.depot.location, patient.location)
                route_slack = max(route.depot.max_work_time - max(route.total_time, route.current_time), 0.0)
                weight = (1.0 / (1.0 + dist)) * (1.0 + _normalize(route_slack, time_scale))
                snapshot.add_edge(patient_key, route_key, "waiting_route", weight)
                snapshot.add_edge(route_key, patient_key, "waiting_route", weight)

            for depot in environment.depots:
                if patient.candidate_centers and depot.id not in patient.candidate_centers:
                    continue
                depot_key = f"depot:{depot.id}"
                dist = environment._travel_time(depot.location, patient.location)
                weight = 1.0 / (1.0 + dist)
                snapshot.add_edge(patient_key, depot_key, "waiting_depot", weight)
                snapshot.add_edge(depot_key, patient_key, "waiting_depot", weight)

        return snapshot

    def _candidate_routes(self, environment: DMDHHRP_Environment, patient: DynamicPatient) -> List[Route]:
        if not patient.candidate_centers:
            return list(environment.routes)
        return [route for route in environment.routes if route.depot.id in patient.candidate_centers]

    def _route_key_for(self, route: Route) -> str:
        return f"route:{route.depot.id}"

    def _patient_key_for(self, patient: Patient) -> str:
        return f"patient:{patient.id}"

    def _urgency_score(self, patient: DynamicPatient, current_time: float) -> float:
        elapsed = max(0.0, current_time - patient.arrival_time)
        if patient.max_tolerance_time <= 0:
            return 1.0
        return 1.0 - _clamp(elapsed / patient.max_tolerance_time, 0.0, 1.0)

    def _route_action_features(
        self,
        environment: DMDHHRP_Environment,
        route: Route,
        patient: DynamicPatient,
        position: int,
        delta_cost: float,
        delta_satisfaction: float,
        current_time: float,
        snapshot: GraphSnapshot,
    ) -> Vector:
        distance = environment._travel_time(route.depot.location, patient.location)
        urgency = self._urgency_score(patient, current_time)
        route_len = len(route.patients)
        route_slack = max(route.depot.max_work_time - max(route.total_time, route.current_time), 0.0)
        candidate_ratio = _normalize(len(patient.candidate_centers), max(len(environment.depots), 1))
        return [
            _normalize(delta_cost, 100.0),
            _normalize(delta_satisfaction, 10.0),
            urgency,
            _normalize(distance, snapshot.xy_scale),
            _normalize(route_len, max(len(environment.known_patients), 1)),
            _normalize(route_slack, snapshot.time_scale),
            _normalize(position, max(route_len + 1, 1)),
            candidate_ratio,
            _normalize(patient.max_tolerance_time, snapshot.time_scale),
            _normalize(patient.service_time, snapshot.time_scale),
        ]

    def _enumerate_insertion_candidates(
        self,
        environment: DMDHHRP_Environment,
        patient: DynamicPatient,
        snapshot: GraphSnapshot,
        embeddings: Dict[str, Vector],
        current_time: float,
    ) -> List[Dict[str, Any]]:
        patient_emb = embeddings.get(self._patient_key_for(patient))
        if patient_emb is None:
            return []

        candidate_routes = self._candidate_routes(environment, patient)
        if not candidate_routes:
            candidate_routes = list(environment.routes)

        candidate_depots = [depot for depot in environment.depots if not patient.candidate_centers or depot.id in patient.candidate_centers]
        route_pool: List[Tuple[Route, bool]] = []
        for route in candidate_routes:
            route_pool.append((route, False))
        for depot in candidate_depots:
            if all(route.depot.id != depot.id for route in environment.routes):
                route_pool.append((Route(depot), True))

        candidates: List[Dict[str, Any]] = []
        for route, is_virtual in route_pool:
            route_key = self._route_key_for(route)
            route_emb = embeddings.get(route_key) or embeddings.get(f"depot:{route.depot.id}")
            if route_emb is None:
                continue

            positions = [0] if is_virtual else list(range(len(route.patients) + 1))
            for pos in positions:
                if pos < len(route.frozen_nodes):
                    continue
                feasible, _ = environment.simulate_route(route, patient, pos)
                if not feasible:
                    continue

                eval_res = environment.evaluate_insertion(route, patient, pos)
                action_features = self._route_action_features(
                    environment=environment,
                    route=route,
                    patient=patient,
                    position=pos,
                    delta_cost=float(eval_res.get("delta_cost", 0.0)),
                    delta_satisfaction=float(eval_res.get("delta_satisfaction", 0.0)),
                    current_time=current_time,
                    snapshot=snapshot,
                )
                forward = self._action_forward(patient_emb, route_emb, action_features)
                score = float(forward["score"])
                score += 0.20 * _normalize(float(eval_res.get("delta_satisfaction", 0.0)), 10.0)
                score -= 0.10 * _normalize(float(eval_res.get("delta_cost", 0.0)), 100.0)
                score += 0.25 * self._urgency_score(patient, current_time)
                score += 0.05 * _normalize(len(route.patients) + 1, max(len(environment.known_patients), 1))

                heuristic_score = (
                    0.20 * _normalize(float(eval_res.get("delta_satisfaction", 0.0)), 10.0)
                    - 0.10 * _normalize(float(eval_res.get("delta_cost", 0.0)), 100.0)
                    + 0.25 * self._urgency_score(patient, current_time)
                    + 0.05 * _normalize(len(route.patients) + 1, max(len(environment.known_patients), 1))
                )

                candidates.append(
                    {
                        "route": route,
                        "is_virtual": is_virtual,
                        "position": pos,
                        "score": score,
                        "heuristic_score": heuristic_score,
                        "patient_emb": list(patient_emb),
                        "route_emb": list(route_emb),
                        "features": list(action_features),
                        "forward": forward,
                    }
                )

        return candidates

    def _update_action_head(self, candidates: Sequence[Dict[str, Any]], target_index: int, learning_rate: float) -> float:
        if not candidates:
            return 0.0

        scores = [float(candidate["score"]) for candidate in candidates]
        probs = self._softmax(scores)
        target_prob = max(probs[target_index], 1e-12)
        loss = -math.log(target_prob)

        grad_v = [0.0 for _ in range(self.action_hidden_dim)]
        grad_v_bias = 0.0
        grad_w_patient = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.action_hidden_dim)]
        grad_w_route = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.action_hidden_dim)]
        grad_w_feat = [[0.0 for _ in range(10)] for _ in range(self.action_hidden_dim)]
        grad_b = [0.0 for _ in range(self.action_hidden_dim)]

        for idx, candidate in enumerate(candidates):
            diff = probs[idx] - (1.0 if idx == target_index else 0.0)
            patient_emb = candidate["patient_emb"]
            route_emb = candidate["route_emb"]
            features = candidate["features"]
            hidden = candidate["forward"]["h"]

            for j in range(self.action_hidden_dim):
                grad_v[j] += diff * hidden[j]
            grad_v_bias += diff

            for j in range(self.action_hidden_dim):
                upstream = diff * self.action_v[j] * (1.0 - hidden[j] * hidden[j])
                grad_b[j] += upstream
                for k in range(self.hidden_dim):
                    grad_w_patient[j][k] += upstream * float(patient_emb[k])
                    grad_w_route[j][k] += upstream * float(route_emb[k])
                for k in range(10):
                    grad_w_feat[j][k] += upstream * float(features[k])

        def _apply_update(matrix: List[List[float]], grad: List[List[float]]) -> None:
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    update = learning_rate * grad[i][j]
                    row[j] = value - _clamp(update, -5.0, 5.0)

        def _apply_vector(vector: List[float], grad: List[float]) -> None:
            for i, value in enumerate(vector):
                update = learning_rate * grad[i]
                vector[i] = value - _clamp(update, -5.0, 5.0)

        _apply_update(self.action_w_patient, grad_w_patient)
        _apply_update(self.action_w_route, grad_w_route)
        _apply_update(self.action_w_feat, grad_w_feat)
        _apply_vector(self.action_b, grad_b)
        _apply_vector(self.action_v, grad_v)
        self.action_v_bias -= _clamp(learning_rate * grad_v_bias, -5.0, 5.0)
        return loss

    def train_episode(
        self,
        environment: DMDHHRP_Environment,
        learning_rate: float = 0.01,
    ) -> Dict[str, float]:
        """对单个环境进行一轮在线模仿训练。"""
        environment.routes = []
        environment._build_initial_solution()

        total_loss = 0.0
        total_decisions = 0
        served_dynamic_ids: List[int] = []

        events = list(environment.dynamic_patients_pool)
        waiting_dynamic: List[DynamicPatient] = []

        for event in events:
            environment.update_time(event.arrival_time)
            environment.freeze_prefixes(environment.current_time)
            waiting_dynamic.append(event)

            still_waiting: List[DynamicPatient] = []
            for patient in waiting_dynamic:
                if environment.current_time - patient.arrival_time > patient.max_tolerance_time:
                    environment.rejected_dynamic.add(patient.id)
                else:
                    still_waiting.append(patient)
            waiting_dynamic = still_waiting

            snapshot = self._build_snapshot(environment, waiting_dynamic, environment.current_time)
            embeddings = self.encoder.encode(snapshot)

            def patient_priority(patient: DynamicPatient) -> Tuple[float, float, float]:
                patient_emb = embeddings.get(self._patient_key_for(patient), _zeros(self.hidden_dim))
                graph_priority = sum(patient_emb) / max(len(patient_emb), 1)
                urgency = self._urgency_score(patient, environment.current_time)
                slack = patient.max_tolerance_time - max(0.0, environment.current_time - patient.arrival_time)
                return (-urgency, -graph_priority, slack)

            for patient in sorted(waiting_dynamic, key=patient_priority):
                if patient.id in environment.rejected_dynamic:
                    continue
                if any(patient.id == p.id for route in environment.routes for p in route.patients):
                    continue

                candidates = self._enumerate_insertion_candidates(environment, patient, snapshot, embeddings, environment.current_time)
                if not candidates:
                    continue

                target_index = max(range(len(candidates)), key=lambda idx: candidates[idx]["heuristic_score"])
                total_loss += self._update_action_head(candidates, target_index, learning_rate)
                total_decisions += 1

                chosen = candidates[target_index]
                if self._insert_patient(environment, patient, chosen["route"], int(chosen["position"])):
                    served_dynamic_ids.append(patient.id)
                    snapshot = self._build_snapshot(environment, waiting_dynamic, environment.current_time)
                    embeddings = self.encoder.encode(snapshot)
                    waiting_dynamic = [p for p in waiting_dynamic if p.id != patient.id]

        obj1, obj2 = environment.evaluate_objectives()
        self.last_served_dynamic_ids = served_dynamic_ids
        return {
            "loss": total_loss,
            "decisions": float(total_decisions),
            "obj1": obj1,
            "obj2": obj2,
            "served_dynamic": float(len(served_dynamic_ids)),
        }

    def fit(
        self,
        environments: Sequence[DMDHHRP_Environment],
        epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> List[Dict[str, float]]:
        """对一组环境重复训练，返回每轮统计。"""
        history: List[Dict[str, float]] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_decisions = 0.0
            epoch_served = 0.0
            epoch_obj1 = 0.0
            epoch_obj2 = 0.0

            for environment in environments:
                stats = self.train_episode(environment, learning_rate=learning_rate)
                epoch_loss += float(stats["loss"])
                epoch_decisions += float(stats["decisions"])
                epoch_served += float(stats["served_dynamic"])
                epoch_obj1 += float(stats["obj1"])
                epoch_obj2 += float(stats["obj2"])

            n_env = max(len(environments), 1)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": epoch_loss / n_env,
                    "decisions": epoch_decisions / n_env,
                    "served_dynamic": epoch_served / n_env,
                    "obj1": epoch_obj1 / n_env,
                    "obj2": epoch_obj2 / n_env,
                }
            )
        return history

    def _select_best_insertion(
        self,
        environment: DMDHHRP_Environment,
        patient: DynamicPatient,
        snapshot: GraphSnapshot,
        embeddings: Dict[str, Vector],
        current_time: float,
    ) -> Tuple[float, Optional[Route], int]:
        best_score = -float("inf")
        best_route: Optional[Route] = None
        best_pos = -1
        candidates = self._enumerate_insertion_candidates(environment, patient, snapshot, embeddings, current_time)
        for candidate in candidates:
            score = float(candidate["score"])
            if score > best_score:
                best_score = score
                best_route = candidate["route"]
                best_pos = int(candidate["position"])
        return best_score, best_route, best_pos

    def _insert_patient(self, environment: DMDHHRP_Environment, patient: DynamicPatient, route: Route, position: int) -> bool:
        route.patients.insert(position, patient)
        feasible, start_times = environment.simulate_route(route)
        if feasible:
            route.service_start_times = start_times
            route.sync_planning_state()
            patient.is_served = True
            if all(existing_route.depot.id != route.depot.id for existing_route in environment.routes):
                environment.routes.append(route)
            return True

        route.patients.pop(position)
        return False

    def _snapshot_solution(self, environment: DMDHHRP_Environment, waiting_dynamic: Sequence[DynamicPatient]) -> Solution:
        assigned_patient_ids = {patient.id for route in environment.routes for patient in route.patients}
        waiting_ids = {patient.id for patient in waiting_dynamic}
        unassigned = {patient.id for patient in environment.known_patients if patient.id not in assigned_patient_ids}
        rejected_dynamic = set(environment.rejected_dynamic)
        for patient in waiting_dynamic:
            if patient.id not in assigned_patient_ids and (environment.current_time - patient.arrival_time) > patient.max_tolerance_time:
                rejected_dynamic.add(patient.id)

        return Solution(
            routes={route.depot.id: route for route in environment.routes},
            unassigned_patients=unassigned,
            accepted_dynamic={patient.id for route in environment.routes for patient in route.patients if patient.is_dynamic},
            rejected_dynamic=rejected_dynamic,
            obj1=0.0,
            obj2=0.0,
        )

    def reoptimize(
        self,
        current_solution: Solution,
        waiting_dynamic: Sequence[DynamicPatient],
        current_time: float,
        environment: DMDHHRP_Environment,
    ) -> Solution:
        """在线重调度：用 GNN 风格打分后选择插入动作。"""
        del current_solution  # 保留接口兼容；当前实现直接基于 environment 状态求解。

        waiting_pool = [patient for patient in waiting_dynamic if patient.id not in environment.rejected_dynamic]
        if not waiting_pool:
            self.last_snapshot = self._build_snapshot(environment, waiting_pool, current_time)
            self.last_embeddings = self.encoder.encode(self.last_snapshot)
            return self._snapshot_solution(environment, waiting_pool)

        served_dynamic_ids: List[int] = []

        # 先构建一版图并得到患者优先级，然后按“更紧急、更容易插入”的顺序处理。
        snapshot = self._build_snapshot(environment, waiting_pool, current_time)
        embeddings = self.encoder.encode(snapshot)

        def patient_priority(patient: DynamicPatient) -> Tuple[float, float, float]:
            patient_emb = embeddings.get(self._patient_key_for(patient), _zeros(self.hidden_dim))
            graph_priority = sum(patient_emb) / max(len(patient_emb), 1)
            urgency = self._urgency_score(patient, current_time)
            slack = patient.max_tolerance_time - max(0.0, current_time - patient.arrival_time)
            return (-urgency, -graph_priority, slack)

        for patient in sorted(waiting_pool, key=patient_priority):
            if patient.id in environment.rejected_dynamic:
                continue
            if any(patient.id == p.id for route in environment.routes for p in route.patients):
                continue

            score, route, position = self._select_best_insertion(environment, patient, snapshot, embeddings, current_time)
            if route is None or position < 0:
                continue
            if score < self.accept_threshold:
                continue

            if self._insert_patient(environment, patient, route, position):
                served_dynamic_ids.append(patient.id)
                # 插入后重建图，让后续动态患者看到新的路径状态。
                snapshot = self._build_snapshot(environment, waiting_pool, current_time)
                embeddings = self.encoder.encode(snapshot)

        if served_dynamic_ids:
            merged = list(self.last_served_dynamic_ids)
            for pid in served_dynamic_ids:
                if pid not in merged:
                    merged.append(pid)
            self.last_served_dynamic_ids = merged
        self.last_snapshot = snapshot
        self.last_embeddings = embeddings
        return self._snapshot_solution(environment, waiting_pool)


__all__ = [
    "GraphGuidedDynamicSolver",
    "GraphNode",
    "GraphReoptimizationResult",
    "GraphSnapshot",
    "PurePythonGraphEncoder",
]