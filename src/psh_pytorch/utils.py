from typing import Union

import torch


C1 = [827, 887, 433, 673, 431,
      271, 587, 607, 859, 953]


def ravel_multi_index(multi_index, dims):
    """Convert a multidimensional index into a single index.

    Args:
        multi_index (torch.Tensor): The multidimensional index as (N, D) tensor.
        dims (torch.Tensor): The dimensions of the multidimensional index.

    Returns:
        torch.Tensor: The single index as (N,) tensor.
    """
    dims_cumprod = torch.tensor(
        dims[:0:-1], device=multi_index.device).cumprod(0)
    return torch.sum(multi_index[..., :-1] * dims_cumprod, dim=-1) + multi_index[..., -1]


def unravel_index(index, dims):
    """Convert a single index into a multidimensional index.

    Args:
        index (torch.Tensor): The single index as (N,) tensor.
        dims (torch.Tensor): The dimensions of the multidimensional index.

    Returns:
        torch.Tensor: The multidimensional index as (N, D) tensor.
    """
    dims_cumprod = torch.tensor(
        dims[:0:-1], device=index.device).cumprod(0)
    multi_index = torch.zeros(
        (*index.shape, len(dims)), device=index.device, dtype=torch.long)
    for i, dim in enumerate(dims_cumprod):
        multi_index[..., i] = index // dim
        index = index % dim
    multi_index[..., -1] = index
    return multi_index


def sparsity_hash(k: Union[int, torch.Tensor], p: torch.Tensor) -> torch.Tensor:
    """Hash function used for sparsity encoding

    Args:
        k: Which hash to use.
        p: Points to hash as (N, D) tensor.

    Returns:
        torch.Tensor: The hash as (N,) tensor.
    """
    if isinstance(k, int):
        k = torch.tensor(k, device=p.device, dtype=torch.uint8)
    d = p.shape[-1]
    k = k.float().unsqueeze(-1)
    p = p.float()
    primes_tensor = torch.tensor(C1[:d], device=p.device, dtype=torch.float)
    hk = (p * (p + k * primes_tensor).rsqrt()).sum(dim=-1).frac()
    return (256 * hk).clamp(0, 255).to(torch.uint8)
