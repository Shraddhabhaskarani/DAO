"""
Drop-in replacement for torch_scatter using PyG's native CUDA-compatible implementations.
Replaces: torch_scatter.scatter, torch_scatter.composite.scatter_softmax
"""

from torch_geometric.utils import scatter as _pyg_scatter
from torch_geometric.utils import softmax as _pyg_softmax


def scatter(src, index, dim=0, dim_size=None, reduce='sum', **kwargs):
    """Drop-in replacement for torch_scatter.scatter using PyG."""
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def scatter_softmax(src, index, dim=0, **kwargs):
    """Drop-in replacement for torch_scatter.composite.scatter_softmax using PyG."""
    return _pyg_softmax(src, index, dim=dim)
