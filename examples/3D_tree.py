import os

import open3d as o3d

def load_tree():
    tree_file = os.path.join(os.path.dirname(__file__), 'data', 'Tree-2.glb')

    mesh = o3d.io.read_triangle_mesh(tree_file)
    print(mesh)

    # Create a visualizer
    vis = o3d.visualization.Visualizer()

    # Add the point cloud to the visualizer
    vis.add_geometry(mesh)

    # Render the visualizer to a png file
    vis.run(o3d.visualization.RenderScene())

    # Save the png file
    with open("my_image.png", "wb") as f:
        f.write(vis.get_render_to_png())

if __name__=='__main__':
    load_tree()