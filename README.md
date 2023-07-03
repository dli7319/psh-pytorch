# Perfect Spatial Hashing in PyTorch
This library is an unofficial implementation of [Perfect Spatial Hashing by Lefebvre and Hoppe](https://hhoppe.com/perfecthash.pdf).

## Usage
Install the library with:
```bash
# For the latest version:
pip install git+https://github.com/dli7319/psh-pytorch.git

# For the PyPI version:
# pip install psh-pytorch
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

## Limitations

1. Currently, interpolation is not supported. Hence, you must use long indices and there are no gradients with respect to the indices.

## Acknowledgement
If you find this library useful, please consider citing the original paper:
```bibtex
@article{lefebvre2006perfect,
author = {Lefebvre, Sylvain and Hoppe, Hugues},
title = {Perfect Spatial Hashing},
year = {2006},
issue_date = {July 2006},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {25},
number = {3},
issn = {0730-0301},
url = {https://doi.org/10.1145/1141911.1141926},
doi = {10.1145/1141911.1141926},
journal = {ACM Trans. Graph.},
month = {jul},
pages = {579–588},
numpages = {10},
keywords = {vector images, minimal perfect hash, sparse data, adaptive textures, multidimensional hashing, 3D-parameterized textures}
}
```