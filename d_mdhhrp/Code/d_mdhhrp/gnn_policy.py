"""PyTorch 版 GNN 策略：动态图编码 + 患者优先级 + 中心匹配打分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - package/script dual use
    from .graph_builder import GraphData
except ImportError:  # pragma: no cover
    from graph_builder import GraphData


class GraphConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(self.update_mlp(torch.cat([h, torch.zeros_like(h)], dim=-1)))

        src, dst = edge_index
        msg_input = torch.cat([h[src], h[dst], edge_features], dim=-1)
        messages = self.message_mlp(msg_input)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)

        degree = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
        ones = torch.ones(dst.size(0), 1, device=h.device, dtype=h.dtype)
        degree.index_add_(0, dst, ones)
        agg = agg / degree.clamp(min=1.0)

        updated = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(updated)


@dataclass
class DispatchOutput:
    node_embeddings: torch.Tensor
    priority_logits: torch.Tensor
    assignment_logits: torch.Tensor
    assignment_scores: torch.Tensor
    patient_indices: torch.Tensor
    depot_indices: torch.Tensor
    pair_index: torch.Tensor


class GNNDispatchPolicy(nn.Module):
    """输出患者优先级和患者-中心匹配分数的轻量 GNN。"""

    def __init__(self, node_dim: int = 16, edge_dim: int = 4, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([GraphConvBlock(hidden_dim, edge_dim) for _ in range(num_layers)])
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.assignment_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, graph: GraphData) -> torch.Tensor:
        h = self.node_encoder(graph.node_features)
        for conv in self.convs:
            h = conv(h, graph.edge_index, graph.edge_features)
        return h

    def forward(self, graph: GraphData) -> DispatchOutput:
        node_embeddings = self.encode(graph)

        if graph.patient_indices.numel() == 0:
            priority_logits = torch.zeros((0,), device=node_embeddings.device, dtype=node_embeddings.dtype)
        else:
            priority_logits = self.priority_head(node_embeddings[graph.patient_indices]).squeeze(-1)

        if graph.pair_index.numel() == 0:
            assignment_scores = torch.zeros((0,), device=node_embeddings.device, dtype=node_embeddings.dtype)
            assignment_logits = torch.zeros(
                (graph.patient_indices.numel(), graph.depot_indices.numel()),
                device=node_embeddings.device,
                dtype=node_embeddings.dtype,
            )
        else:
            pair_src = graph.pair_index[:, 0]
            pair_dst = graph.pair_index[:, 1]
            pair_input = torch.cat(
                [
                    node_embeddings[pair_src],
                    node_embeddings[pair_dst],
                    graph.pair_features,
                ],
                dim=-1,
            )
            assignment_scores = self.assignment_head(pair_input).squeeze(-1)
            assignment_logits = torch.full(
                (graph.patient_indices.numel(), graph.depot_indices.numel()),
                -1e9,
                device=node_embeddings.device,
                dtype=node_embeddings.dtype,
            )
            patient_local_map = {int(node_idx): i for i, node_idx in enumerate(graph.patient_indices.tolist())}
            depot_local_map = {int(node_idx): i for i, node_idx in enumerate(graph.depot_indices.tolist())}
            for pair_idx, (patient_node_idx, depot_node_idx) in enumerate(graph.pair_index.tolist()):
                patient_local_idx = patient_local_map.get(int(patient_node_idx))
                depot_local_idx = depot_local_map.get(int(depot_node_idx))
                if patient_local_idx is None or depot_local_idx is None:
                    continue
                assignment_logits[patient_local_idx, depot_local_idx] = assignment_scores[pair_idx]

        return DispatchOutput(
            node_embeddings=node_embeddings,
            priority_logits=priority_logits,
            assignment_logits=assignment_logits,
            assignment_scores=assignment_scores,
            patient_indices=graph.patient_indices,
            depot_indices=graph.depot_indices,
            pair_index=graph.pair_index,
        )

    @torch.no_grad()
    def predict(self, graph: GraphData) -> Dict[str, torch.Tensor]:
        output = self.forward(graph)
        return {
            "priority_logits": output.priority_logits,
            "assignment_logits": output.assignment_logits,
            "assignment_scores": output.assignment_scores,
            "patient_indices": output.patient_indices,
            "depot_indices": output.depot_indices,
            "pair_index": output.pair_index,
        }


__all__ = ["DispatchOutput", "GNNDispatchPolicy", "GraphConvBlock"]