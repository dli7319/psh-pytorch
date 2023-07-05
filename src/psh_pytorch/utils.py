from typing import Union, Sequence

import torch


C1 = [79, 23, 71, 31, 11, 64, 41, 67, 89, 43]


def ravel_multi_index(multi_index, dims):
    """Convert a multidimensional index into a single index.

    Args:
        multi_index (torch.Tensor): The multidimensional index as (N, D) tensor.
        dims (torch.Tensor): The dimensions of the multidimensional index.

    Returns:
        torch.Tensor: The single index as (N,) tensor.
    """
    dims_cumprod = torch.tensor(
        tuple(dims[1:]) + (1,), device=multi_index.device)
    dims_cumprod = dims_cumprod.flip(0).cumprod(0).flip(0)
    return torch.sum(multi_index * dims_cumprod, dim=-1)


def unravel_index(index, dims):
    """Convert a single index into a multidimensional index.

    Args:
        index (torch.Tensor): The single index as (N,) tensor.
        dims (torch.Tensor): The dimensions of the multidimensional index.

    Returns:
        torch.Tensor: The multidimensional index as (N, D) tensor.
    """
    dims_cumprod = torch.tensor(tuple(dims) + (1, ), device=index.device)
    dims_cumprod = dims_cumprod.flip(0).cumprod(0).flip(0)
    return index.unsqueeze(-1) % dims_cumprod[:-1] // dims_cumprod[1:]


def sparsity_hash(k: Union[int, torch.Tensor], p: torch.Tensor,
                  primes: Union[Sequence[int], torch.Tensor] = C1) -> torch.Tensor:
    """Hash function used for sparsity encoding

    Args:
        k: Which hash to use.
        p: Points to hash as (N, D) tensor.
        primes: (Optional) The primes to use for the hash.

    Returns:
        torch.Tensor: The hash as (N,) tensor.
    """
    k = torch.as_tensor(k, device=p.device, dtype=torch.float32)
    k = k.reshape((-1,) * (p.dim() - 1) + (1,))
    p = p.float()
    primes = (torch.as_tensor(primes[:p.shape[-1]])
              .to(device=p.device, dtype=torch.float))
    hk = (p * (p + k * primes).rsqrt()).sum(dim=-1).frac()
    return (256 * hk).clamp(0, 255).to(torch.uint8)
