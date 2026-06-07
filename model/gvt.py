
from typing import Tuple, Optional, Any, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, bias=True):
        super().__init__()
        if n_layers == 1:
            self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=bias)])
        else:
            dims = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
            self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(n_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
            nn.init.orthogonal_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, x: Tensor) -> Tensor:
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.gelu(x)
        x = self.lins[-1](x)
        return x


class GraphViewTransformation(nn.Module):
    def __init__(self, adj_max_hop: int = 2, vt_depth: int = 2):
        super().__init__()
        self.adj_max_hop = adj_max_hop
        self.vt_depth = vt_depth
        self.in_C = self.hid_C = 1 + (adj_max_hop * 2)

        self._init_parameters()
        self.reset_parameters()

    def _init_parameters(self):
        self.mlp = MLP(self.in_C, self.hid_C, 1, self.vt_depth)

    def reset_parameters(self):
        with torch.no_grad():
            self.mlp.reset_parameters()

    @staticmethod
    def _build_adjacency_matrices(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
        device, dtype = edge_index.device, torch.float32
        row, col = edge_index
        weights = torch.ones(row.size(0), dtype=dtype, device=device)
        deg = torch.zeros(num_nodes, dtype=dtype, device=device)
        deg.scatter_add_(0, row, weights)
        deg_inv = torch.where(deg != 0, 1.0 / deg, torch.zeros_like(deg))
        rw_weights = deg_inv[row] * weights
        A_rw = torch.sparse_coo_tensor(torch.stack([row, col]), rw_weights, (num_nodes, num_nodes))
        sym_index, sym_weight = gcn_norm(edge_index, None, num_nodes, False, True, 'source_to_target', dtype)
        A_sym = torch.sparse_coo_tensor(sym_index, sym_weight, (num_nodes, num_nodes))
        return A_rw, A_sym

    def _view_stacking(self, x: Tensor, A_rw: Optional[Tensor] = None, A_sym: Optional[Tensor] = None) -> Tensor:
        N, Fdim = x.shape
        num_views = 1 + self.adj_max_hop * 2
        views = x.new_empty(N, Fdim, num_views)
        v = 0
        # Initial view (Identity)
        views[..., v] = x; v += 1
        y_rw = y_sym = x
        # A_RW and A_SYM views
        for _ in range(1, self.adj_max_hop + 1):
            # Random Walk
            y_rw = torch.sparse.mm(A_rw, y_rw)
            views[..., v] = y_rw; v += 1
            # Symmetric
            y_sym = torch.sparse.mm(A_sym, y_sym)
            views[..., v] = y_sym; v += 1
        return views

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None, A_rw: Optional[Tensor] = None, 
                A_sym: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        
        if (A_rw is None or A_sym is None) and edge_index is not None:
            A_rw, A_sym = self._build_adjacency_matrices(edge_index, x.size(0))
        views = self._view_stacking(x, A_rw, A_sym)

        # Apply the shared view-vector mapping phi.
        N, Fdim, C = views.shape  # [N, F, C]
        x_reshaped = views.reshape(-1, C)  # [N*F, C]
        x = self.mlp(x_reshaped)  # [N*F, 1]
        x = x.view(N, Fdim)  # [N, F]

        return x
