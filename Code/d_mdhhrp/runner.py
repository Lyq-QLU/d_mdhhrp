"""一个轻量的本地运行入口，便于把新架构快速串起来。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - package/script dual use
    from .data_loader import HybridInstance, build_environment, generate_random_hybrid_instance, hybrid_instance_to_dict, load_hybrid_instance_from_json, load_solomon_dynamic_instance, load_solomon_instance, save_hybrid_instance_to_json
    from .dispatch import dispatch_dynamic_patients
    from .gnn_policy import GNNDispatchPolicy
    from .hybrid_solver import HybridRollingHorizonSolver
    from .gnn_solver import GraphGuidedDynamicSolver
    from .simulator import DynamicSchedulingSimulator
except ImportError:  # pragma: no cover
    from data_loader import HybridInstance, build_environment, generate_random_hybrid_instance, hybrid_instance_to_dict, load_hybrid_instance_from_json, load_solomon_dynamic_instance, load_solomon_instance, save_hybrid_instance_to_json
    from dispatch import dispatch_dynamic_patients
    from gnn_policy import GNNDispatchPolicy
    from hybrid_solver import HybridRollingHorizonSolver
    from gnn_solver import GraphGuidedDynamicSolver
    from simulator import DynamicSchedulingSimulator

def run_random_demo(
    num_centers: int = 3,
    num_patients: int = 20,
    dynamic_ratio: float = 0.3,
    seed: Optional[int] = None,
):
    instance = generate_random_hybrid_instance(
        num_centers=num_centers,
        num_patients=num_patients,
        dynamic_ratio=dynamic_ratio,
        area_size=20.0,
        seed=seed,
    )
    return run_instance(instance)


def run_instance(instance: HybridInstance, offline_solver: Optional[Any] = None, online_solver: Optional[Any] = None):
    environment = build_environment(instance)
    simulator = DynamicSchedulingSimulator(environment)
    return simulator.run(offline_solver=offline_solver, online_solver=online_solver)


def run_gnn_demo(
    num_centers: int = 3,
    num_patients: int = 20,
    dynamic_ratio: float = 0.3,
    seed: Optional[int] = None,
):
    instance = generate_random_hybrid_instance(
        num_centers=num_centers,
        num_patients=num_patients,
        dynamic_ratio=dynamic_ratio,
        area_size=20.0,
        seed=seed,
    )
    solver = GraphGuidedDynamicSolver()
    return run_instance(instance, online_solver=solver)


def run_training_demo(
    epochs: int = 3,
    episodes_per_epoch: int = 4,
    num_centers: int = 3,
    num_patients: int = 12,
    dynamic_ratio: float = 0.33,
    seed: int = 7,
    learning_rate: float = 0.01,
    dataset_dir: str = '/Users/marc/Documents/Code/dataset/main',
    limit: Optional[int] = None,
):
    from .train import TrainConfig, train_supervised_gnn

    _ = (episodes_per_epoch, num_centers, num_patients, dynamic_ratio)
    config = TrainConfig(
        dataset_dir=dataset_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        limit=limit,
    )
    return train_supervised_gnn(config)


def run_hybrid_demo(
    num_centers: int = 3,
    num_patients: int = 20,
    dynamic_ratio: float = 0.3,
    seed: Optional[int] = None,
    local_search_iters: int = 3,
    destroy_ratio: float = 0.25,
):
    instance = generate_random_hybrid_instance(
        num_centers=num_centers,
        num_patients=num_patients,
        dynamic_ratio=dynamic_ratio,
        area_size=20.0,
        seed=seed,
    )
    solver = HybridRollingHorizonSolver(
        seed=seed or 7,
        local_search_iters=local_search_iters,
        destroy_ratio=destroy_ratio,
    )
    return run_instance(instance, online_solver=solver)


def run_gnn_dispatch_demo(
    num_centers: int = 3,
    num_patients: int = 20,
    dynamic_ratio: float = 0.3,
    seed: Optional[int] = None,
    top_k_centers: int = 2,
    alns_rounds: int = 2,
):
    instance = generate_random_hybrid_instance(
        num_centers=num_centers,
        num_patients=num_patients,
        dynamic_ratio=dynamic_ratio,
        area_size=20.0,
        seed=seed,
    )
    environment = build_environment(instance)
    policy = GNNDispatchPolicy()
    return dispatch_dynamic_patients(
        environment,
        policy=policy,
        waiting_dynamic=environment.dynamic_patients_pool,
        top_k_centers=top_k_centers,
        alns_rounds=alns_rounds,
    )


def run_solomon_dynamic_demo(
    file_path: str,
    preset_center_ids: Optional[list[int]] = None,
    dynamic_ratio: float = 0.3,
    dynamic_release_strategy: str = "midpoint",
    dynamic_arrival_mode: str = "strategy",
    center_generation: str = "preset",
    num_centers: Optional[int] = None,
    seed: int = 7,
    top_k_centers: int = 2,
    alns_rounds: int = 2,
):
    instance = load_solomon_dynamic_instance(
        file_path,
        preset_center_ids=preset_center_ids,
        dynamic_ratio=dynamic_ratio,
        dynamic_release_strategy=dynamic_release_strategy,
        dynamic_arrival_mode=dynamic_arrival_mode,
        center_generation=center_generation,
        num_centers=num_centers,
        seed=seed,
    )
    environment = build_environment(instance)
    policy = GNNDispatchPolicy()
    return dispatch_dynamic_patients(
        environment,
        policy=policy,
        waiting_dynamic=environment.dynamic_patients_pool,
        top_k_centers=top_k_centers,
        alns_rounds=alns_rounds,
    )


def run_solomon_dynamic_export_demo(
    file_path: str,
    output_path: str,
    preset_center_ids: Optional[list[int]] = None,
    dynamic_ratio: float = 0.3,
    dynamic_release_strategy: str = "midpoint",
    dynamic_arrival_mode: str = "strategy",
    center_generation: str = "preset",
    num_centers: Optional[int] = None,
    seed: int = 7,
):
    instance = load_solomon_dynamic_instance(
        file_path,
        preset_center_ids=preset_center_ids,
        dynamic_ratio=dynamic_ratio,
        dynamic_release_strategy=dynamic_release_strategy,
        dynamic_arrival_mode=dynamic_arrival_mode,
        center_generation=center_generation,
        num_centers=num_centers,
        seed=seed,
    )
    save_hybrid_instance_to_json(instance, output_path)
    reloaded = load_hybrid_instance_from_json(output_path)
    return instance, reloaded


def run_batch_dataset_generation(
    solomon_dir: str = '/Users/marc/Documents/Code/in',
    output_dir: str = '/Users/marc/Documents/Code/dataset',
    num_files: Optional[int] = None,
    verbose: bool = True,
):
    """Generate the current benchmark dataset layout for training/evaluation."""
    from .dataset_batch_generator import generate_dataset

    return generate_dataset(
        solomon_dir=solomon_dir,
        output_dir=output_dir,
        num_seeds=5,
        verbose=verbose,
    )


def run_main_experiment_batch(
    dataset_dir: str = '/Users/marc/Documents/Code/dataset/main',
    output_file: str = '/Users/marc/Documents/Code/results/main_experiment_summary.json',
    solver_type: str = 'hybrid',  # 'hybrid' or 'gnn'
    solver_seed: int = 7,
    local_search_iters: int = 5,
    destroy_ratio: float = 0.25,
    limit: Optional[int] = None,
):
    """Run the main experiment over the generated main benchmark split.
    
    Args:
        dataset_dir: Root directory containing instance .json files
        output_file: Where to write results summary
        solver_type: 'hybrid' (ALNS) or 'gnn' (Graph-guided)
        solver_seed: Random seed for solver
        local_search_iters: ALNS iterations per round
        destroy_ratio: ALNS destruction ratio
        limit: Limit number of instances (None = all)
    """
    dataset_root = Path(dataset_dir)
    json_files = [p for p in sorted(dataset_root.rglob('*.json')) if p.name != 'metadata.json']
    if limit is not None:
        json_files = json_files[:limit]

    results = []
    for json_path in json_files:
        instance = load_hybrid_instance_from_json(json_path)
        
        if solver_type == 'gnn':
            # Use GNN-guided dispatch
            try:
                from .dispatch import dispatch_dynamic_patients
                from .gnn_policy import GNNDispatchPolicy
                
                environment = build_environment(instance)
                policy = GNNDispatchPolicy()
                dispatch_result = dispatch_dynamic_patients(
                    environment,
                    policy=policy,
                    waiting_dynamic=environment.dynamic_patients_pool,
                    top_k_centers=2,
                    alns_rounds=local_search_iters,
                )
                solution = dispatch_result.solution
            except Exception as e:
                print(f"[GNN] Error on {json_path.name}: {e}")
                continue
        else:
            # Use hybrid ALNS
            solver = HybridRollingHorizonSolver(
                seed=solver_seed,
                local_search_iters=local_search_iters,
                destroy_ratio=destroy_ratio,
            )
            result = run_instance(instance, online_solver=solver)
            solution = result.solution
            
        results.append({
            'file': str(json_path),
            'solver': solver_type,
            'obj1': float(solution.obj1),
            'obj2': float(solution.obj2),
            'unassigned': len(solution.unassigned_patients),
            'accepted_dynamic': len(solution.accepted_dynamic),
            'rejected_dynamic': len(solution.rejected_dynamic),
        })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    result = run_gnn_dispatch_demo()
    print("solution obj1:", result.solution.obj1)
    print("solution obj2:", result.solution.obj2)
    print("unassigned:", len(result.solution.unassigned_patients))
