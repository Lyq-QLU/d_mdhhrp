"""数据加载与混合实例生成。

当前问题定义：
------------------------------------------------------------
已知：
1. 一组服务中心位置。
2. 一组患者（客户），每个患者有服务时间窗、服务时长以及动态到达时间。
3. 路网为欧氏距离。

决策变量：
1. 患者如何分配到已有中心。
2. 每辆车的服务路径。

目标函数：
1. 最小化总成本（主要是路径成本）。

约束：
1. 每个患者被唯一分配到一个中心且被服务。
2. 路径时间窗与动态到达约束满足。

Solomon 实例映射：
1. depot 作为中心节点。
2. 其他点作为患者。
3. 可通过参数指定哪些节点作为中心。
4. 可将部分患者按比例转换为动态患者，并在后续时刻释放到调度环境中。
------------------------------------------------------------
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - package/script dual use
    from .models import Depot, DynamicPatient, Location, ScheduledPatient, DMDHHRP_Environment
except ImportError:  # pragma: no cover
    from models import Depot, DynamicPatient, Location, ScheduledPatient, DMDHHRP_Environment


@dataclass
class HybridInstance:
    centers: List[Depot]
    scheduled_patients: List[ScheduledPatient]
    dynamic_patients: List[DynamicPatient]
    travel_speed: float = 1.0
    circle_threshold: float = 15.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_patients(self) -> List:
        return [*self.scheduled_patients, *self.dynamic_patients]


def validate_hybrid_instance(
    instance: HybridInstance,
    *,
    expected_total_patients: Optional[int] = None,
    forbid_patient_id_zero: bool = True,
) -> None:
    """校验混合实例基本一致性，失败时抛出 ValueError。"""

    errors: List[str] = []
    all_patients = [*instance.scheduled_patients, *instance.dynamic_patients]
    total_patients = len(all_patients)

    if expected_total_patients is not None and total_patients != int(expected_total_patients):
        errors.append(
            f"total_patients mismatch: got {total_patients}, expected {int(expected_total_patients)}"
        )

    if forbid_patient_id_zero and any(int(p.id) == 0 for p in all_patients):
        errors.append("patient id 0 detected (Solomon depot should not be treated as patient)")

    seen_ids = set()
    dup_ids = set()
    for p in all_patients:
        pid = int(p.id)
        if pid in seen_ids:
            dup_ids.add(pid)
        seen_ids.add(pid)
    if dup_ids:
        errors.append(f"duplicate patient ids: {sorted(dup_ids)}")

    for p in instance.scheduled_patients:
        if float(p.service_time) < 0:
            errors.append(f"scheduled patient {p.id}: negative service_time")
        if float(p.hard_tw[0]) > float(p.hard_tw[1]):
            errors.append(f"scheduled patient {p.id}: invalid hard_tw {p.hard_tw}")

    day_start = min((float(p.hard_tw[0]) for p in all_patients), default=0.0)
    for p in instance.dynamic_patients:
        if float(p.service_time) < 0:
            errors.append(f"dynamic patient {p.id}: negative service_time")
        if float(p.arrival_time) < day_start:
            errors.append(
                f"dynamic patient {p.id}: arrival_time {p.arrival_time} < day_start {day_start}"
            )

    if not instance.centers:
        errors.append("no centers generated")

    center_ids = [int(c.id) for c in instance.centers]
    if len(set(center_ids)) != len(center_ids):
        errors.append("duplicate center ids")

    if errors:
        raise ValueError("HybridInstance validation failed: " + " | ".join(errors))


def _depot_to_dict(depot: Depot) -> Dict[str, Any]:
    return {
        "id": int(depot.id),
        "location": {"x": float(depot.location.x), "y": float(depot.location.y)},
        "fixed_cost": float(depot.fixed_cost),
        "max_work_time": float(depot.max_work_time),
    }


def _patient_to_dict(patient: ScheduledPatient | DynamicPatient | Any) -> Dict[str, Any]:
    return {
        "id": int(patient.id),
        "location": {"x": float(patient.location.x), "y": float(patient.location.y)},
        "service_time": float(patient.service_time),
        "is_dynamic": bool(patient.is_dynamic),
        "hard_tw": [float(patient.hard_tw[0]), float(patient.hard_tw[1])],
        "soft_tw": [float(patient.soft_tw[0]), float(patient.soft_tw[1])],
        "arrival_time": float(getattr(patient, "arrival_time", 0.0)),
        "ideal_response_time": float(getattr(patient, "ideal_response_time", 15.0)),
        "max_tolerance_time": float(getattr(patient, "max_tolerance_time", 60.0)),
        "circle_id": int(getattr(patient, "circle_id", -1)),
        "candidate_centers": [int(cid) for cid in getattr(patient, "candidate_centers", [])],
    }


def hybrid_instance_to_dict(instance: HybridInstance) -> Dict[str, Any]:
    """把混合实例序列化成 JSON 友好的字典。"""

    return {
        "format": "d_mdhhrp_hybrid_instance",
        "centers": [_depot_to_dict(depot) for depot in instance.centers],
        "scheduled_patients": [_patient_to_dict(patient) for patient in instance.scheduled_patients],
        "dynamic_patients": [_patient_to_dict(patient) for patient in instance.dynamic_patients],
        "travel_speed": float(instance.travel_speed),
        "circle_threshold": float(instance.circle_threshold),
        "metadata": dict(instance.metadata),
    }


def save_hybrid_instance_to_json(instance: HybridInstance, file_path: str | Path, indent: int = 2) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_instance_to_dict(instance), f, ensure_ascii=False, indent=indent)


def _patient_from_dict(patient_data: Dict[str, Any]) -> ScheduledPatient | DynamicPatient:
    location = _as_point(patient_data.get("location", [0.0, 0.0]))
    hard_tw = patient_data.get("hard_tw") or [0.0, float("inf")]
    soft_tw = patient_data.get("soft_tw") or hard_tw
    common_kwargs = dict(
        id=int(patient_data.get("id", 0)),
        location=location,
        service_time=float(patient_data.get("service_time", 0.0)),
        hard_tw=(float(hard_tw[0]), float(hard_tw[1])),
        soft_tw=(float(soft_tw[0]), float(soft_tw[1])),
        arrival_time=float(patient_data.get("arrival_time", 0.0)),
        ideal_response_time=float(patient_data.get("ideal_response_time", 15.0)),
        max_tolerance_time=float(patient_data.get("max_tolerance_time", 60.0)),
        circle_id=int(patient_data.get("circle_id", -1)),
        candidate_centers=[int(cid) for cid in patient_data.get("candidate_centers", [])],
    )
    if bool(patient_data.get("is_dynamic", False)):
        return DynamicPatient(is_dynamic=True, **common_kwargs)
    return ScheduledPatient(is_dynamic=False, **common_kwargs)


def _as_point(value: Any) -> Location:
    if isinstance(value, Location):
        return value
    if isinstance(value, dict):
        return Location(float(value.get("x", 0.0)), float(value.get("y", 0.0)))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return Location(float(value[0]), float(value[1]))
    raise TypeError(f"Unsupported location format: {value!r}")


def _build_soft_window_from_hard(hard_start: float, hard_end: float, shrink_ratio: float = 0.6) -> Tuple[float, float]:
    width = max(0.0, hard_end - hard_start)
    if width <= 0:
        return hard_start, hard_start + 1e-3
    inner = width * shrink_ratio
    margin = (width - inner) / 2.0
    soft_start = hard_start + margin
    soft_end = hard_end - margin
    if soft_end <= soft_start:
        soft_end = soft_start + 1e-3
    return soft_start, soft_end


def _normalize_ids(values: Optional[Sequence[int]], valid_ids: Sequence[int]) -> Optional[List[int]]:
    if values is None:
        return None
    valid_set = set(valid_ids)
    normalized: List[int] = []
    for value in values:
        cid = int(value)
        if cid in valid_set and cid not in normalized:
            normalized.append(cid)
    return normalized


def _build_dynamic_release_time(
    hard_start: float,
    hard_end: float,
    ideal_response_time: float,
    max_tolerance_time: float,
    rng: random.Random,
    strategy: str = "midpoint",
) -> float:
    """根据时间窗与策略生成动态患者释放时刻。"""

    window_width = max(hard_end - hard_start, 1.0)
    ideal = float(ideal_response_time if ideal_response_time > 0 else 0.25 * window_width)
    max_resp = max(float(max_tolerance_time), ideal + 0.1, 0.5 * window_width)

    strategy = strategy.lower().strip()
    if strategy == "early":
        lead = rng.uniform(0.7 * max_resp, max_resp)
    elif strategy == "late":
        lead = rng.uniform(0.1 * max_resp, 0.4 * max_resp)
    elif strategy == "uniform":
        lead = rng.uniform(0.1 * window_width, max_resp)
    else:
        lead = rng.uniform(0.25 * max_resp, 0.85 * max_resp)

    reveal_time = max(hard_start, (hard_start + hard_end) / 2.0 - lead)
    return float(reveal_time)


def _kmeans_center_locations(
    points: Sequence[Location],
    k: int,
    rng: random.Random,
    max_iters: int = 30,
) -> List[Location]:
    """纯 Python K-means，返回 k 个中心点坐标。"""

    if not points:
        return [Location(0.0, 0.0) for _ in range(max(1, k))]

    k = max(1, min(int(k), len(points)))
    init_indices = rng.sample(range(len(points)), k)
    centroids = [Location(points[i].x, points[i].y) for i in init_indices]

    for _ in range(max_iters):
        clusters: List[List[Location]] = [[] for _ in range(k)]
        for pt in points:
            best_idx = min(
                range(k),
                key=lambda idx: (pt.x - centroids[idx].x) ** 2 + (pt.y - centroids[idx].y) ** 2,
            )
            clusters[best_idx].append(pt)

        new_centroids: List[Location] = []
        for idx, cluster in enumerate(clusters):
            if not cluster:
                fallback = points[rng.randrange(len(points))]
                new_centroids.append(Location(fallback.x, fallback.y))
                continue
            cx = sum(p.x for p in cluster) / len(cluster)
            cy = sum(p.y for p in cluster) / len(cluster)
            new_centroids.append(Location(cx, cy))

        moved = sum(
            (centroids[i].x - new_centroids[i].x) ** 2 + (centroids[i].y - new_centroids[i].y) ** 2
            for i in range(k)
        )
        centroids = new_centroids
        if moved < 1e-8:
            break

    return centroids


def _split_patients_into_hybrid(
    centers: Sequence[Depot],
    raw_patients: Sequence[ScheduledPatient],
    dynamic_ratio: float,
    rng: random.Random,
    circle_threshold: float,
    dynamic_release_strategy: str = "midpoint",
    dynamic_arrival_mode: str = "strategy",
    day_start: Optional[float] = None,
    day_end: Optional[float] = None,
) -> Tuple[List[ScheduledPatient], List[DynamicPatient]]:
    """将静态患者列表按比例拆成预约患者和动态患者。"""

    if not raw_patients:
        return [], []

    dyn_count = max(0, min(len(raw_patients), int(round(len(raw_patients) * max(0.0, dynamic_ratio)))))
    dynamic_idx = set(rng.sample(range(len(raw_patients)), dyn_count)) if dyn_count > 0 else set()

    scheduled_patients: List[ScheduledPatient] = []
    dynamic_patients: List[DynamicPatient] = []

    for idx, patient in enumerate(raw_patients):
        build_candidate_centers(centers, patient, threshold=circle_threshold)
        if idx in dynamic_idx:
            hard_start, hard_end = patient.hard_tw
            mode = (dynamic_arrival_mode or "strategy").strip().lower()
            if mode == "ready_time":
                reveal_time = float(hard_start)
            elif mode == "uniform_day":
                low = float(day_start if day_start is not None else 0.0)
                high = float(day_end if day_end is not None else max(hard_end, low + 1.0))
                if high <= low:
                    high = low + 1.0
                reveal_time = float(rng.uniform(low, high))
            else:
                reveal_time = _build_dynamic_release_time(
                    hard_start=hard_start,
                    hard_end=hard_end,
                    ideal_response_time=patient.ideal_response_time,
                    max_tolerance_time=patient.max_tolerance_time,
                    rng=rng,
                    strategy=dynamic_release_strategy,
                )
            dynamic_patients.append(
                DynamicPatient(
                    id=patient.id,
                    location=patient.location,
                    service_time=patient.service_time,
                    is_dynamic=True,
                    hard_tw=patient.hard_tw,
                    soft_tw=patient.soft_tw,
                    arrival_time=reveal_time,
                    ideal_response_time=patient.ideal_response_time,
                    max_tolerance_time=patient.max_tolerance_time,
                    circle_id=patient.circle_id,
                    candidate_centers=list(patient.candidate_centers),
                )
            )
        else:
            scheduled_patients.append(
                ScheduledPatient(
                    id=patient.id,
                    location=patient.location,
                    service_time=patient.service_time,
                    is_dynamic=False,
                    hard_tw=patient.hard_tw,
                    soft_tw=patient.soft_tw,
                    arrival_time=patient.arrival_time,
                    ideal_response_time=patient.ideal_response_time,
                    max_tolerance_time=patient.max_tolerance_time,
                    circle_id=patient.circle_id,
                    candidate_centers=list(patient.candidate_centers),
                )
            )

    scheduled_patients.sort(key=lambda p: p.id)
    dynamic_patients.sort(key=lambda p: p.arrival_time)
    return scheduled_patients, dynamic_patients


def build_candidate_centers(centers: Sequence[Depot], patient, threshold: float = 15.0, allow_fallback: bool = True) -> List[int]:
    candidate_centers: List[int] = []
    seen_ids = set()
    for depot in centers:
        if depot.id in seen_ids:
            continue
        seen_ids.add(depot.id)
        travel_time = depot.location.distance_to(patient.location)
        if travel_time <= threshold:
            candidate_centers.append(depot.id)
    if not candidate_centers and centers and allow_fallback:
        nearest = min(centers, key=lambda depot: depot.location.distance_to(patient.location))
        candidate_centers = [nearest.id]
    patient.candidate_centers = candidate_centers
    return candidate_centers


def _parse_center(center_data: Dict[str, Any], default_id: int) -> Depot:
    center_id = int(center_data.get("id", default_id))
    location = _as_point(center_data.get("location", [0.0, 0.0]))
    fixed_cost = float(center_data.get("fixed_cost", center_data.get("center_fixed_cost", 100.0)))
    max_work_time = float(center_data.get("max_work_time", center_data.get("due_date", 12.0)))
    return Depot(center_id, location, fixed_cost, max_work_time)


def _parse_patient(patient_data: Dict[str, Any], default_id: int):
    pid = int(patient_data.get("id", default_id))
    location = _as_point(patient_data.get("location", [0.0, 0.0]))
    service_time = float(patient_data.get("service_time", 0.0))

    hard_window = patient_data.get("hard_window") or [patient_data.get("ready_time", 0.0), patient_data.get("due_date", 0.0)]
    hard_start, hard_end = float(hard_window[0]), float(hard_window[1])
    soft_window = patient_data.get("soft_window")
    if soft_window is None:
        soft_window = _build_soft_window_from_hard(hard_start, hard_end)
    soft_start, soft_end = float(soft_window[0]), float(soft_window[1])

    is_dynamic = bool(patient_data.get("is_dynamic", False))
    arrival_time = float(patient_data.get("arrival_time", patient_data.get("reveal_time", 0.0)))
    ideal_response = float(patient_data.get("ideal_response_time", patient_data.get("ideal_response", 15.0)))
    max_response = float(
        patient_data.get(
            "max_tolerance_time",
            patient_data.get("max_response", max(ideal_response + 1e-3, hard_end - hard_start)),
        )
    )

    common_kwargs = dict(
        id=pid,
        location=location,
        service_time=service_time,
        hard_tw=(hard_start, hard_end),
        soft_tw=(soft_start, soft_end),
        arrival_time=arrival_time,
        ideal_response_time=ideal_response,
        max_tolerance_time=max_response,
        circle_id=int(patient_data.get("circle_id", -1)),
    )

    if is_dynamic:
        return DynamicPatient(is_dynamic=True, **common_kwargs)
    return ScheduledPatient(is_dynamic=False, **common_kwargs)


def load_solomon_instance(
    file_path: str | Path,
    *,
    center_ids: Optional[Sequence[int]] = None,
    preset_center_ids: Optional[Sequence[int]] = None,
    coverage_threshold: float = 15.0,
    vehicle_count: Optional[int] = None,
    vehicle_capacity: Optional[float] = None,
) -> HybridInstance:
    """加载 Solomon 格式实例，默认将 depot 节点作为中心，其余节点作为患者。"""
    lines: List[str] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())

    vehicle_section = False
    customer_section = False
    vehicle_num: Optional[int] = None
    vehicle_cap: Optional[float] = None
    customers: List[Dict[str, float]] = []

    for line in lines:
        upper = line.upper()
        if upper.startswith("VEHICLE"):
            vehicle_section = True
            customer_section = False
            continue
        if upper.startswith("CUSTOMER"):
            customer_section = True
            vehicle_section = False
            continue
        if vehicle_section and ("CAPACITY" in upper or "NUMBER" in upper):
            continue
        if vehicle_section and not customer_section:
            parts = line.split()
            if len(parts) >= 2:
                vehicle_num = int(parts[0])
                vehicle_cap = float(parts[1])
            continue
        if customer_section:
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                customers.append(
                    {
                        "id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "demand": float(parts[3]),
                        "ready_time": float(parts[4]),
                        "due_date": float(parts[5]),
                        "service_time": float(parts[6]),
                    }
                )

    if vehicle_count is not None:
        vehicle_num = vehicle_count
    if vehicle_capacity is not None:
        vehicle_cap = vehicle_capacity

    active_ids = list(preset_center_ids if preset_center_ids is not None else (center_ids if center_ids is not None else [0]))
    active_ids = _normalize_ids(active_ids, [c["id"] for c in customers]) or [0]
    centers: List[Depot] = []
    for cid in active_ids:
        c = customers[cid]
        centers.append(
            Depot(
                id=cid,
                location=Location(c["x"], c["y"]),
                fixed_cost=0.0,
                max_work_time=c["due_date"],
            )
        )

    scheduled_patients: List[ScheduledPatient] = []
    for c in customers:
        if c["id"] in active_ids:
            continue
        soft_tw = _build_soft_window_from_hard(c["ready_time"], c["due_date"])
        scheduled_patients.append(
            ScheduledPatient(
                id=c["id"],
                location=Location(c["x"], c["y"]),
                service_time=c["service_time"],
                hard_tw=(c["ready_time"], c["due_date"]),
                soft_tw=soft_tw,
                candidate_centers=[],
            )
        )

    for patient in scheduled_patients:
        build_candidate_centers(centers, patient, threshold=coverage_threshold)

    metadata = dict(
        vehicle_num=vehicle_num,
        vehicle_cap=vehicle_cap,
        file=str(file_path),
        center_ids=active_ids,
        preset_center_ids=active_ids,
        coverage_threshold=coverage_threshold,
    )

    return HybridInstance(
        centers=centers,
        scheduled_patients=scheduled_patients,
        dynamic_patients=[],
        travel_speed=1.0,
        circle_threshold=coverage_threshold,
        metadata=metadata,
    )


def load_solomon_dynamic_instance(
    file_path: str | Path,
    *,
    center_ids: Optional[Sequence[int]] = None,
    preset_center_ids: Optional[Sequence[int]] = None,
    dynamic_ratio: float = 0.3,
    dynamic_release_strategy: str = "midpoint",
    dynamic_arrival_mode: str = "strategy",
    center_generation: str = "preset",
    num_centers: Optional[int] = None,
    num_patients: Optional[int] = None,
    seed: Optional[int] = None,
    coverage_threshold: float = 15.0,
    travel_speed: float = 1.0,
    reserve_ratio: float = 0.2,
    vehicle_count: Optional[int] = None,
    vehicle_capacity: Optional[float] = None,
) -> HybridInstance:
    """加载 Solomon 实例，并将其中一部分患者转换为动态患者。"""

    lines: List[str] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())

    vehicle_section = False
    customer_section = False
    vehicle_num: Optional[int] = None
    vehicle_cap: Optional[float] = None
    customers: List[Dict[str, float]] = []

    for line in lines:
        upper = line.upper()
        if upper.startswith("VEHICLE"):
            vehicle_section = True
            customer_section = False
            continue
        if upper.startswith("CUSTOMER"):
            customer_section = True
            vehicle_section = False
            continue
        if vehicle_section and ("CAPACITY" in upper or "NUMBER" in upper):
            continue
        if vehicle_section and not customer_section:
            parts = line.split()
            if len(parts) >= 2:
                vehicle_num = int(parts[0])
                vehicle_cap = float(parts[1])
            continue
        if customer_section:
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                customers.append(
                    {
                        "id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "demand": float(parts[3]),
                        "ready_time": float(parts[4]),
                        "due_date": float(parts[5]),
                        "service_time": float(parts[6]),
                    }
                )

    if vehicle_count is not None:
        vehicle_num = vehicle_count
    if vehicle_capacity is not None:
        vehicle_cap = vehicle_capacity

    rng = random.Random(seed)
    depot_candidates = [c for c in customers if int(c.get("id", -1)) == 0]
    depot_record = depot_candidates[0] if depot_candidates else (customers[0] if customers else None)
    patient_customers = [c for c in customers if int(c.get("id", -1)) != int(depot_record.get("id", -1))] if depot_record is not None else list(customers)
    
    if num_patients is not None:
        patient_customers = patient_customers[:num_patients]
        
    center_mode = (center_generation or "preset").strip().lower()

    centers: List[Depot] = []
    active_ids: List[int] = []

    if center_mode == "kmeans":
        k = int(num_centers) if num_centers is not None else 3
        point_list = [Location(c["x"], c["y"]) for c in patient_customers]
        centroid_locs = _kmeans_center_locations(point_list, k=k, rng=rng)
        horizon = max((float(c["due_date"]) for c in patient_customers), default=12.0)
        centers = [
            Depot(
                id=idx,
                location=loc,
                fixed_cost=0.0,
                max_work_time=horizon,
            )
            for idx, loc in enumerate(centroid_locs)
        ]
    else:
        active_ids = list(preset_center_ids if preset_center_ids is not None else (center_ids if center_ids is not None else [0]))
        active_ids = _normalize_ids(active_ids, [c["id"] for c in customers]) or [0]
        for cid in active_ids:
            matched = next((item for item in customers if int(item["id"]) == int(cid)), None)
            c = matched if matched is not None else customers[0]
            centers.append(
                Depot(
                    id=cid,
                    location=Location(c["x"], c["y"]),
                    fixed_cost=0.0,
                    max_work_time=c["due_date"],
                )
            )

    raw_patients: List[ScheduledPatient] = []
    for c in patient_customers:
        if center_mode != "kmeans" and c["id"] in active_ids:
            continue
        soft_tw = _build_soft_window_from_hard(c["ready_time"], c["due_date"])
        raw_patients.append(
            ScheduledPatient(
                id=c["id"],
                location=Location(c["x"], c["y"]),
                service_time=c["service_time"],
                hard_tw=(c["ready_time"], c["due_date"]),
                soft_tw=soft_tw,
                candidate_centers=[],
            )
        )

    day_start = min((float(c["ready_time"]) for c in patient_customers), default=0.0)
    day_end = max((float(c["due_date"]) for c in patient_customers), default=day_start + 1.0)
    scheduled_patients, dynamic_patients = _split_patients_into_hybrid(
        centers=centers,
        raw_patients=raw_patients,
        dynamic_ratio=dynamic_ratio,
        rng=rng,
        circle_threshold=coverage_threshold,
        dynamic_release_strategy=dynamic_release_strategy,
        dynamic_arrival_mode=dynamic_arrival_mode,
        day_start=day_start,
        day_end=day_end,
    )

    metadata = dict(
        vehicle_num=vehicle_num,
        vehicle_cap=vehicle_cap,
        file=str(file_path),
        center_ids=active_ids,
        preset_center_ids=active_ids,
        coverage_threshold=coverage_threshold,
        dynamic_ratio=dynamic_ratio,
        dynamic_release_strategy=dynamic_release_strategy,
        dynamic_arrival_mode=dynamic_arrival_mode,
        center_generation=center_mode,
        num_centers=len(centers),
        seed=seed,
    )

    instance = HybridInstance(
        centers=centers,
        scheduled_patients=scheduled_patients,
        dynamic_patients=dynamic_patients,
        travel_speed=travel_speed,
        circle_threshold=coverage_threshold,
        metadata=metadata,
    )
    validate_hybrid_instance(instance, expected_total_patients=len(patient_customers), forbid_patient_id_zero=True)
    return instance


def make_hybrid_instance(
    centers_data: Sequence[Dict[str, Any]],
    patients_data: Sequence[Dict[str, Any]],
    dynamic_ratio: float = 0.3,
    seed: Optional[int] = None,
    travel_speed: float = 1.0,
    circle_threshold: float = 15.0,
    reserve_ratio: float = 0.2,
) -> HybridInstance:
    rng = random.Random(seed)

    centers = [_parse_center(center, idx) for idx, center in enumerate(sorted(centers_data, key=lambda c: int(c.get("id", 0))))]
    if not centers:
        centers = [Depot(0, Location(0.0, 0.0), 100.0, 12.0)]

    raw_patients = [_parse_patient(patient, idx) for idx, patient in enumerate(sorted(patients_data, key=lambda p: int(p.get("id", 0))))]
    if not raw_patients:
        return HybridInstance(
            centers=centers,
            scheduled_patients=[],
            dynamic_patients=[],
            travel_speed=travel_speed,
            circle_threshold=circle_threshold,
        )

    dyn_count = max(0, min(len(raw_patients), int(round(len(raw_patients) * dynamic_ratio))))
    dynamic_idx = set(rng.sample(range(len(raw_patients)), dyn_count))

    scheduled_patients: List[ScheduledPatient] = []
    dynamic_patients: List[DynamicPatient] = []

    for idx, patient in enumerate(raw_patients):
        build_candidate_centers(centers, patient, threshold=circle_threshold)
        if idx in dynamic_idx:
            hard_start, hard_end = patient.hard_tw
            window_width = max(hard_end - hard_start, 1.0)
            ideal = float(patient.ideal_response_time if patient.ideal_response_time > 0 else 0.25 * window_width)
            if ideal <= 0:
                ideal = max(0.1, 0.25 * window_width)
            max_resp = max(float(patient.max_tolerance_time), ideal + 0.1, 0.5 * window_width)
            lead = rng.uniform(0.1 * window_width, max_resp)
            reveal_time = max(hard_start, (hard_start + hard_end) / 2.0 - lead)
            dynamic_patients.append(
                DynamicPatient(
                    id=patient.id,
                    location=patient.location,
                    service_time=patient.service_time,
                    is_dynamic=True,
                    hard_tw=patient.hard_tw,
                    soft_tw=patient.soft_tw,
                    arrival_time=reveal_time,
                    ideal_response_time=ideal,
                    max_tolerance_time=max_resp,
                    circle_id=patient.circle_id,
                    candidate_centers=list(patient.candidate_centers),
                )
            )
        else:
            scheduled_patients.append(
                ScheduledPatient(
                    id=patient.id,
                    location=patient.location,
                    service_time=patient.service_time,
                    is_dynamic=False,
                    hard_tw=patient.hard_tw,
                    soft_tw=patient.soft_tw,
                    arrival_time=patient.arrival_time,
                    ideal_response_time=patient.ideal_response_time,
                    max_tolerance_time=patient.max_tolerance_time,
                    circle_id=patient.circle_id,
                    candidate_centers=list(patient.candidate_centers),
                )
            )

    scheduled_patients.sort(key=lambda p: p.id)
    dynamic_patients.sort(key=lambda p: p.arrival_time)
    return HybridInstance(
        centers=centers,
        scheduled_patients=scheduled_patients,
        dynamic_patients=dynamic_patients,
        travel_speed=travel_speed,
        circle_threshold=circle_threshold,
        metadata={
            "reserve_ratio": reserve_ratio,
            "dynamic_ratio": dynamic_ratio,
            "seed": seed,
        },
    )


def load_hybrid_instance_from_json(
    file_path: str | Path,
    dynamic_ratio: float = 0.3,
    seed: Optional[int] = None,
    travel_speed: float = 1.0,
    circle_threshold: float = 15.0,
    reserve_ratio: float = 0.2,
) -> HybridInstance:
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("format") == "d_mdhhrp_hybrid_instance" or (
        "scheduled_patients" in payload and "dynamic_patients" in payload
    ):
        centers = [
            Depot(
                id=int(center.get("id", idx)),
                location=_as_point(center.get("location", [0.0, 0.0])),
                fixed_cost=float(center.get("fixed_cost", 100.0)),
                max_work_time=float(center.get("max_work_time", 12.0)),
            )
            for idx, center in enumerate(payload.get("centers", []))
        ]
        scheduled_patients = [_patient_from_dict(item) for item in payload.get("scheduled_patients", [])]
        dynamic_patients = [_patient_from_dict(item) for item in payload.get("dynamic_patients", [])]
        for patient in scheduled_patients + dynamic_patients:
            if not patient.candidate_centers:
                build_candidate_centers(centers, patient, threshold=payload.get("circle_threshold", circle_threshold))
        return HybridInstance(
            centers=centers,
            scheduled_patients=scheduled_patients,
            dynamic_patients=dynamic_patients,
            travel_speed=float(payload.get("travel_speed", travel_speed)),
            circle_threshold=float(payload.get("circle_threshold", circle_threshold)),
            metadata=dict(payload.get("metadata", {})),
        )

    centers_data = payload.get("centers", [])
    patients_data = payload.get("patients", [])
    return make_hybrid_instance(
        centers_data=centers_data,
        patients_data=patients_data,
        dynamic_ratio=dynamic_ratio,
        seed=seed,
        travel_speed=travel_speed,
        circle_threshold=circle_threshold,
        reserve_ratio=reserve_ratio,
    )


def build_environment(instance: HybridInstance, **kwargs) -> DMDHHRP_Environment:
    params = dict(
        depots=instance.centers,
        scheduled_patients=instance.scheduled_patients,
        dynamic_patients=instance.dynamic_patients,
        R_threshold=kwargs.pop("R_threshold", instance.circle_threshold),
        travel_speed=kwargs.pop("travel_speed", instance.travel_speed),
        reserve_ratio=kwargs.pop("reserve_ratio", instance.metadata.get("reserve_ratio", 0.2)),
    )
    params.update(kwargs)
    return DMDHHRP_Environment(**params)


def generate_random_hybrid_instance(
    num_centers: int = 3,
    num_patients: int = 20,
    area_size: float = 20.0,
    dynamic_ratio: float = 0.3,
    seed: Optional[int] = None,
    travel_speed: float = 1.0,
    circle_threshold: float = 15.0,
) -> HybridInstance:
    rng = random.Random(seed)

    centers_data: List[Dict[str, Any]] = []
    for i in range(num_centers):
        centers_data.append(
            {
                "id": i,
                "location": [rng.uniform(0, area_size), rng.uniform(0, area_size)],
                "fixed_cost": 100.0,
                "max_work_time": 100.0,
            }
        )

    patients_data: List[Dict[str, Any]] = []
    for i in range(num_patients):
        hard_start = rng.uniform(8.0, 12.0)
        hard_end = rng.uniform(hard_start + 2.0, 20.0)
        soft_start, soft_end = _build_soft_window_from_hard(hard_start, hard_end, shrink_ratio=0.6)
        patients_data.append(
            {
                "id": i,
                "location": [rng.uniform(0, area_size), rng.uniform(0, area_size)],
                "service_time": rng.uniform(0.2, 0.5),
                "hard_window": [hard_start, hard_end],
                "soft_window": [soft_start, soft_end],
            }
        )

    return make_hybrid_instance(
        centers_data=centers_data,
        patients_data=patients_data,
        dynamic_ratio=dynamic_ratio,
        seed=seed,
        travel_speed=travel_speed,
        circle_threshold=circle_threshold,
    )
