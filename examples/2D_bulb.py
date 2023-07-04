import os

import torch
from PIL import Image
import numpy as np
import tqdm

# Typically, you can import from psh_pytorch directly.
from src.psh_pytorch import PerfectSpatialHash


def encode_bulb():
    """Encode the bulb image into the hash table and reconstruct it."""
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

    spatial_hash = PerfectSpatialHash(
        bulb_occupancy_grid, 3, offset_table_size=18,
        verbose=False)

    indices = torch.stack(torch.meshgrid(torch.arange(128, device=device),
                                         torch.arange(128, device=device),
                                         indexing='ij'), -1)

    # Save the image into the hash table.
    valid_pixels = bulb_occupancy_grid.nonzero(as_tuple=False).long()
    assert not spatial_hash.check_collisions(
        valid_pixels), "There are collisions in the hash table."
    spatial_hash.encode(
        valid_pixels, bulb_screenshot[valid_pixels.unbind(-1)][:, :3])

    values, sparsity = spatial_hash(indices.reshape(-1, 2))
    reconstruction = (values * sparsity.unsqueeze(-1)).reshape(128, 128, 3)
    # Save the reconstruction image.
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray((reconstruction.detach().cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "bulb_reconstruction.png"))


def encode_bulb_with_gd():
    """Encode the bulb image using gradient descent and reconstruct it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bulb_screenshot_path = os.path.join(
        os.path.dirname(__file__), 'data', 'bulb.png')

    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'gd')
    os.makedirs(output_dir, exist_ok=True)
    iterations = 100

    bulb_screenshot = np.array(Image.open(
        bulb_screenshot_path), dtype=np.float32) / 255
    bulb_screenshot = torch.from_numpy(bulb_screenshot).to(device)

    height, width, _ = bulb_screenshot.shape
    assert height == width == 128, "Bulb screenshot is not 128x128 pixels."

    bulb_occupancy_grid = bulb_screenshot[:, :, 3]
    assert torch.sum(bulb_occupancy_grid.bool()
                     ) == 1381, "Number of occupied pixels is not correct."

    spatial_hash = PerfectSpatialHash(
        bulb_occupancy_grid, 3, offset_table_size=18,
        verbose=False)

    optimizer = torch.optim.Adam(spatial_hash.parameters(), lr=0.01)

    indices = torch.stack(torch.meshgrid(torch.arange(128, device=device),
                                         torch.arange(128, device=device),
                                         indexing='ij'), -1)

    pbar = tqdm.trange(iterations, desc="GD Iteration")
    for i in pbar:
        optimizer.zero_grad()
        values, sparsity = spatial_hash(indices.reshape(-1, 2))
        reconstruction = (values * sparsity.unsqueeze(-1)).reshape(128, 128, 3)
        loss = torch.sum((reconstruction - bulb_screenshot[:, :, :3]) ** 2)
        pbar.set_postfix({'loss': loss.item()})
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            with torch.no_grad():
                Image.fromarray((reconstruction.detach().clip(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(output_dir, f"bulb_reconstruction_{i:04d}.png"))

    # Save the reconstruction image.
    reconstruction = (values * sparsity.unsqueeze(-1)).reshape(128, 128, 3)
    Image.fromarray((reconstruction.detach().clip(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "bulb_reconstruction_final.png"))

    # Save the hash table.
    hash_table = spatial_hash.hash_table.detach().cpu().numpy()
    Image.fromarray((hash_table * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "bulb_hash_table.png"))

    # Save the offset table.
    offset_table = spatial_hash.offset_table.detach().cpu().numpy()
    offset_table = np.concatenate(
        (offset_table, np.ones_like(offset_table[..., :1])), axis=-1)
    Image.fromarray((offset_table * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "bulb_offset_table.png"))

    # Save the sparsity encoding.
    sparsity_encoding = spatial_hash.sparsity_encoding.detach().cpu().numpy()
    sparsity_encoding = np.concatenate(
        (sparsity_encoding, np.ones_like(sparsity_encoding[..., :1])), axis=-1)
    Image.fromarray((sparsity_encoding * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "bulb_sparsity_encoding.png"))


if __name__ == '__main__':
    # encode_bulb()
    encode_bulb_with_gd()
