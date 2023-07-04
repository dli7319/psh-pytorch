import os

import open3d as o3d
import torch
import matplotlib.pyplot as plt

from src.psh_pytorch import PerfectSpatialHash

def load_tree():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tree_file = os.path.join(os.path.dirname(__file__), 'data', 'Tree-2.glb')

    mesh = o3d.io.read_triangle_mesh(tree_file)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.05)

    voxels = voxel_grid.get_voxels()
    indices = torch.tensor([x.grid_index for x in voxels], dtype=torch.int32, device=device)

    occupancy_grid_resolution = 64
    occupancy_grid = torch.zeros((occupancy_grid_resolution,) * 3, dtype=torch.bool, device=device)
    occupancy_grid[indices.unbind(-1)] = True

    spatial_hash = PerfectSpatialHash(occupancy_grid, 3)
    
    indices = torch.stack(torch.meshgrid(
        torch.arange(0, occupancy_grid_resolution, dtype=torch.int32, device=device),
        torch.arange(0, occupancy_grid_resolution, dtype=torch.int32, device=device),
        torch.arange(0, occupancy_grid_resolution, dtype=torch.int32, device=device),
        indexing='ij'), dim=-1)
        
    values, sparsity = spatial_hash(indices.reshape(-1, 3))

    sparsity = sparsity.reshape(occupancy_grid_resolution, occupancy_grid_resolution, occupancy_grid_resolution)

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(sparsity.detach().cpu().numpy(), edgecolor='k')

    output_dir = os.path.join(os.path.dirname(__file__), 'output', "3D_tree")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sparsity.png"))

if __name__=='__main__':
    load_tree()