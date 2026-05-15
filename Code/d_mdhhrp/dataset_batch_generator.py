"""Batch dataset generation for GNN training on dynamic VRPTW instances based on Ma et al. 2025."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from d_mdhhrp.data_loader import (
    load_solomon_dynamic_instance,
    save_hybrid_instance_to_json,
)

@dataclass
class InstanceConfig:
    solomon_file: str
    num_patients: int
    dynamic_ratio: float
    num_centers: int
    seed: int
    experiment_group: str  # "main" or "sensitivity"
    size_label: str  # "S", "M", "L"
    dist_label: str  # "C", "R", "RC"


def generate_experiment_configs(
    solomon_dir: str,
    num_seeds: int = 3
) -> List[InstanceConfig]:
    configs = []
    
    # Representative instances
    reps = {
        "C": "c101.txt",
        "R": "r101.txt",
        "RC": "rc101.txt"
    }
    
    # 规模对应患者数
    sizes = {"S": 40, "M": 60, "L": 80}
    
    # 一、主实验: 固定动态比例 0.3，不同规模和分布
    for dist_label, file_name in reps.items():
        file_path = os.path.join(solomon_dir, file_name)
        if not os.path.exists(file_path):
            continue
            
        for size_label, num_patients in sizes.items():
            for seed in range(num_seeds):
                configs.append(InstanceConfig(
                    solomon_file=file_path,
                    num_patients=num_patients,
                    dynamic_ratio=0.3,
                    num_centers=3,
                    seed=seed,
                    experiment_group="main",
                    size_label=size_label,
                    dist_label=dist_label
                ))
                
    # 二、敏感性实验: 固定 RC-M, 变动态比例
    rc_file = os.path.join(solomon_dir, "rc101.txt")
    if os.path.exists(rc_file):
        for dynamic_ratio in [0.1, 0.3, 0.5]:
            for seed in range(num_seeds):
                configs.append(InstanceConfig(
                    solomon_file=rc_file,
                    num_patients=60, # M size
                    dynamic_ratio=dynamic_ratio,
                    num_centers=3,
                    seed=seed,
                    experiment_group="sensitivity",
                    size_label="M",
                    dist_label="RC"
                ))

    return configs


def generate_dataset(
    solomon_dir: str,
    output_dir: str,
    num_seeds: int = 3,
    verbose: bool = True,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    configs = generate_experiment_configs(solomon_dir, num_seeds)
    if verbose:
        print(f"[Dataset] Generated {len(configs)} configuration tasks.")
        
    metadata = {
        'total_instances': 0,
        'errors': 0,
        'instances': []
    }
    
    for i, cfg in enumerate(configs):
        try:
            inst = load_solomon_dynamic_instance(
                cfg.solomon_file,
                center_generation='kmeans',
                num_centers=cfg.num_centers,
                num_patients=cfg.num_patients,
                dynamic_ratio=cfg.dynamic_ratio,
                dynamic_arrival_mode='ready_time',
                seed=cfg.seed,
            )
            
            # Sub-directory logic
            ratio_str = f"D{int(cfg.dynamic_ratio*100)}"
            group_dir = output_path / cfg.experiment_group / f"{cfg.dist_label}_{cfg.size_label}_{ratio_str}"
            group_dir.mkdir(parents=True, exist_ok=True)
            
            instance_id = f"{Path(cfg.solomon_file).stem}_{cfg.size_label}_{ratio_str}_s{cfg.seed}"
            output_file = group_dir / f"{instance_id}.json"
            
            save_hybrid_instance_to_json(inst, str(output_file))
            
            metadata['instances'].append({
                'id': instance_id,
                'experiment_group': cfg.experiment_group,
                'dist_label': cfg.dist_label,
                'size_label': cfg.size_label,
                'num_patients': cfg.num_patients,
                'dynamic_ratio': cfg.dynamic_ratio,
                'seed': cfg.seed,
                'file': str(output_file.relative_to(output_path))
            })
            metadata['total_instances'] += 1
            
            if verbose and (i+1) % 5 == 0:
                print(f"[Dataset] Generated {i+1}/{len(configs)} instances...")
                
        except Exception as e:
            if verbose:
                print(f"[Dataset] ERROR on {cfg}: {e}")
            metadata['errors'] += 1
            
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    if verbose:
        print(f"[Dataset] Completed. {metadata['total_instances']} instances generated, {metadata['errors']} errors.")

if __name__ == "__main__":
    generate_dataset(
        solomon_dir='/Users/marc/Documents/Code/in',
        output_dir='/Users/marc/Documents/Code/dataset',
        num_seeds=5
    )

def load_dataset_split(
    output_dir: str,
    split: str = 'train',
) -> List[Tuple[str, Dict]]:
    """Load all generated dataset instances from a split directory.

    Args:
        output_dir: Root output directory produced by `generate_dataset`.
        split: One of 'train', 'val', or 'test'.

    Returns:
        A list of (json_path, instance_dict) tuples.
    """
    split_dir = Path(output_dir) / split
    instances: List[Tuple[str, Dict]] = []

    search_root = split_dir if split_dir.exists() else Path(output_dir)

    for json_file in sorted(search_root.rglob("*.json")):
        if json_file.name == "metadata.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            instances.append((str(json_file), json.load(f)))

    return instances
