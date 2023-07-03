# perfecthash-pytorch
Perfect Spatial Hashing in PyTorch

## Usage
Install the library with:
```bash
pip install git+https://github.com/dli7319/psh-pytorch.git
```

Instantiate a `PerfectSpatialHash` object with:
```python
from psh_pytorch import PerfectSpatialHash

occupancy_grid # A 2D (or higher dimensional) occupancy grid
out_features = 3
perfect_hash = PerfectSpatialHash(
    occupancy_grid, out_features)

# 2D forward pass
indices = torch.stack(torch.meshgrid(
    torch.arange(128, device=device),
    torch.arange(128, device=device),
    indexing='ij'), -1)
values, sparsity = perfect_hash(indices.reshape(-1, 2))
#   Values are of shape (128 * 128, 3)
#   Sparsity is of shape (128 * 128)
#   Note that values are unmasked
masked_values = values * sparsity.unsqueeze(-1)
```

## Examples
See `examples/` for a simple example of how to use the library.