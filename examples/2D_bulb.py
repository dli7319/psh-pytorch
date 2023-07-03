import os

import torch
from PIL import Image
import numpy as np

from psh_pytorch import PerfectSpatialHash


def test_bulb():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bulb_screenshot_path = os.path.join(
        os.path.dirname(__file__), 'data', 'bulb.png')

    bulb_screenshot = np.array(Image.open(
        bulb_screenshot_path), dtype=np.float32) / 255
    bulb_screenshot = torch.from_numpy(bulb_screenshot).to(device)

    height, width, _ = bulb_screenshot.shape
    assert height == width == 128, "Bulb screenshot is not 128x128 pixels."

    bulb_occupancy_grid = bulb_screenshot[:, :, 3]
    assert torch.sum(bulb_occupancy_grid.bool()
                     ) == 1381, "Number of occupied pixels is not correct."

    perfect_hash = PerfectSpatialHash(
        bulb_occupancy_grid, 3, offset_table_size=18,
        verbose=False)

    indices = torch.stack(torch.meshgrid(torch.arange(128, device=device),
                                         torch.arange(128, device=device),
                                         indexing='ij'), -1)

    # Save the image into the hash table.
    valid_pixels = bulb_occupancy_grid.nonzero(as_tuple=False).long()
    assert not perfect_hash.check_collisions(
        valid_pixels), "There are collisions in the hash table."
    perfect_hash.encode(
        valid_pixels, bulb_screenshot[valid_pixels.unbind(-1)][:, :3])

    values, sparsity = perfect_hash(indices.reshape(-1, 2))
    reconstruction = (values * sparsity.unsqueeze(-1)).reshape(128, 128, 3)
    # Save the reconstruction image.
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray((reconstruction.detach().cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "bulb_reconstruction.png"))


if __name__ == '__main__':
    test_bulb()
