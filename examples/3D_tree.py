import os

import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.psh_pytorch import PerfectSpatialHash

def load_tree():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tree_file = os.path.join(os.path.dirname(__file__), 'data', 'Tree-2.glb')

    mesh = o3d.io.read_triangle_mesh(tree_file)
    # print(mesh)

    # # Create a visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # Add the point cloud to the visualizer
    # vis.add_geometry(mesh)

    # # Render the visualizer to a png file
    # vis.run()

    # Save the png file
    # with open("my_image.png", "wb") as f:
    #     f.write(vis.get_render_to_png())
    print("mesh", mesh)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.05)
    print("voxel_grid", voxel_grid)
    # o3d.visualization.draw_geometries([voxel_grid])

    voxels = voxel_grid.get_voxels()
    print("voxels", voxels)
    bbox = voxel_grid.get_axis_aligned_bounding_box()
    print("bbox", bbox)
    print("voxel", voxels[0].grid_index)
    indices = torch.tensor([x.grid_index for x in voxels], dtype=torch.int32, device=device)
    print("indices", indices.shape)
    print("max index", indices.max(0))

    occupancy_grid_resolution = 64
    occupancy_grid = torch.zeros((occupancy_grid_resolution,) * 3, dtype=torch.bool, device=device)
    occupancy_grid[indices.unbind(-1)] = True
    print("Number of voxels", occupancy_grid.sum().item(), occupancy_grid.numel())

    spatial_hash = PerfectSpatialHash(occupancy_grid, 3,
                                      verbose=True)
    
    indices = torch.stack(torch.meshgrid(
        torch.arange(0, occupancy_grid_resolution, dtype=torch.int32, device=device),
        torch.arange(0, occupancy_grid_resolution, dtype=torch.int32, device=device),
        torch.arange(0, occupancy_grid_resolution, dtype=torch.int32, device=device),
        indexing='ij'))
    
    values, sparsity = spatial_hash(indices.reshape(-1, 3))

    sparsity = sparsity.reshape(occupancy_grid_resolution, occupancy_grid_resolution, occupancy_grid_resolution)
    # print("sparsity", sparsity)

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(occupancy_grid.detach().cpu().numpy(), edgecolor='k')

    plt.show()

if __name__=='__main__':
    load_tree()