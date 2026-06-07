import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from model.gvt import GraphViewTransformation
from typing import Optional

def init_graph_view_transformation(args, device='cpu'):
    graph_view_transformation = GraphViewTransformation(
        adj_max_hop=args.adj_max_hop,
        vt_depth=args.vt_depth
    ).to(device)
    return graph_view_transformation

def recurrent_gvt(graph_view_transformation: GraphViewTransformation, x: Tensor, edge_index: Tensor,
                          args = None, manual_depth: int = None, training: bool = False,
                          A_rw: Optional[Tensor] = None, A_sym: Optional[Tensor] = None):
    """Apply a shared Graph View Transformation recurrently."""
    repeats = args.max_depth if manual_depth is None else manual_depth
    x = x.clone()
    
    # Build adjacency matrices if not provided
    if A_rw is None and A_sym is None:
        A_rw, A_sym = GraphViewTransformation._build_adjacency_matrices(edge_index, x.size(0))

    if training:
        # During training, allow gradients to flow
        for i in range(repeats):
            x = graph_view_transformation(x, edge_index=None, A_rw=A_rw, A_sym=A_sym)
    else:
        # During evaluation, disable gradients for efficiency
        with torch.no_grad():
            for i in range(repeats):
                x = graph_view_transformation(x, edge_index=None, A_rw=A_rw, A_sym=A_sym)
    return x

class Predictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, is_mlp: bool = False):
        super().__init__()
        self.is_mlp = is_mlp
        if is_mlp:
            self.mlp = nn.Sequential(
                Linear(in_channels, 128, bias=bias),
                nn.ReLU(),
                Linear(128, out_channels, bias=bias)
            )
        else:
            self.final_linear = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_mlp:
            for m in self.mlp:
                if isinstance(m, Linear):
                    nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            nn.init.orthogonal_(self.final_linear.weight)
            if self.final_linear.bias is not None:
                nn.init.zeros_(self.final_linear.bias)

    def forward(self, x_agg: Tensor) -> Tensor:
        if self.is_mlp:
            return self.mlp(x_agg)
        else:
            return self.final_linear(x_agg)
