"""d_mdhhrp: dynamic multi-center home healthcare routing toolkit."""

from .models import (
    DMDHHRP_Environment,
    Depot,
    DynamicPatient,
    Location,
    Patient,
    PatientType,
    Route,
    ScheduledPatient,
    Solution,
)
from .data_loader import (
    HybridInstance,
    build_candidate_centers,
    build_environment,
    generate_random_hybrid_instance,
    hybrid_instance_to_dict,
    load_hybrid_instance_from_json,
    load_solomon_instance,
    load_solomon_dynamic_instance,
    make_hybrid_instance,
    save_hybrid_instance_to_json,
    validate_hybrid_instance,
)
from .gnn_solver import GraphGuidedDynamicSolver
from .graph_builder import GraphData, build_graph_from_env
from .gnn_policy import GNNDispatchPolicy
from .hybrid_solver import HybridRollingHorizonSolver
from .dispatch import DispatchResult, dispatch_dynamic_patients
from .label_generator import ExpertLabel, generate_expert_label
from .dataset_batch_generator import (
    load_dataset_split,
)
from .solution_converter import (
    build_solution_from_path,
    solution_to_dict,
    solution_to_path,
    validate_solution_from_path,
)

__all__ = [
    "DMDHHRP_Environment",
    "Depot",
    "DynamicPatient",
    "HybridInstance",
    "Location",
    "Patient",
    "PatientType",
    "Route",
    "ScheduledPatient",
    "Solution",
    "build_candidate_centers",
    "build_environment",
    "build_solution_from_path",
    "generate_random_hybrid_instance",
    "hybrid_instance_to_dict",
    "GraphGuidedDynamicSolver",
    "GraphData",
    "build_graph_from_env",
    "GNNDispatchPolicy",
    "HybridRollingHorizonSolver",
    "DispatchResult",
    "dispatch_dynamic_patients",
    "ExpertLabel",
    "generate_expert_label",
    "load_dataset_split",
    "load_hybrid_instance_from_json",
    "load_solomon_instance",
    "load_solomon_dynamic_instance",
    "make_hybrid_instance",
    "save_hybrid_instance_to_json",
    "validate_hybrid_instance",
    "solution_to_dict",
    "solution_to_path",
    "validate_solution_from_path",
]
