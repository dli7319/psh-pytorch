import unittest

from PIL import Image
import numpy as np
import torch
from psh_pytorch import PerfectSpatialHash


class Test2D(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = device = torch.device("cpu")
        bulb_image_path = "examples/data/bulb.png"
        bulb_image = np.array(Image.open(bulb_image_path),
                              dtype=np.float32) / 255.0
        self.bulb_image = torch.from_numpy(bulb_image).to(device)
        self.bulb_occupancy_grid = self.bulb_image[:, :, 3]
        self.bulb_valid_pixels = self.bulb_occupancy_grid.nonzero(
            as_tuple=False).long()

        self.spatial_hash = PerfectSpatialHash(
            self.bulb_image[:, :, 3], 3)

        self.indices = torch.stack(torch.meshgrid(torch.arange(128, device=device),
                                                  torch.arange(
            128, device=device),
            indexing='ij'), -1)

    def test_collisions(self):
        """
        Check that there are no collisions in the hash table.
        """
        self.assertFalse(self.spatial_hash.check_collisions(
            self.bulb_valid_pixels))

    def test_reconstruction(self):
        """
        Test that we can encode and decode the image from the spatial hash.
        """
        self.spatial_hash.encode(
            self.bulb_valid_pixels,
            self.bulb_image[self.bulb_valid_pixels.unbind(-1)][:, :3])
        values, sparsity = self.spatial_hash(self.indices.reshape(-1, 2))

        # Check that the sparsity is correct.
        sparsity_error_string = f"Returned Sparsity: {sparsity.sum()}, Expected Sparsity: {self.bulb_valid_pixels.shape[0]}"
        self.assertTrue(torch.allclose(
            sparsity, self.bulb_image[:, :, 3].bool().reshape(-1)),
            sparsity_error_string)

        reconstruction = (values * sparsity.unsqueeze(-1)).reshape(128, 128, 3)
        gt_image = (self.bulb_image[:, :, :3] *
                    self.bulb_image[:, :, 3].bool().unsqueeze(-1))

        # Check that the reconstruction is correct.
        error_string = f"Max Error: {torch.max(torch.abs(reconstruction - gt_image))}"
        self.assertTrue(torch.allclose(reconstruction, gt_image), error_string)

    def test_gradient_descent(self):
        """
        Test that gradients can propagate to the spatial hash.
        """
        values, sparsity = self.spatial_hash(self.indices.reshape(-1, 2))
        reconstruction = (values * sparsity.unsqueeze(-1)).reshape(128, 128, 3)
        gt_image = (self.bulb_image[:, :, :3] *
                    self.bulb_image[:, :, 3].bool().unsqueeze(-1))
        loss = torch.sum((reconstruction - gt_image)**2)
        loss.backward()
        loss = loss.detach()

        # Check that gradients are non-zero.
        self.assertTrue(torch.any(self.spatial_hash.hash_table.grad != 0),
                        "Gradients are all zero.")

        # Check that the gradients are correct.
        self.spatial_hash.hash_table.data -= 0.01 * self.spatial_hash.hash_table.grad
        values, sparsity = self.spatial_hash(self.indices.reshape(-1, 2))
        reconstruction = (values * sparsity.unsqueeze(-1)).reshape(128, 128, 3)
        new_loss = torch.sum((reconstruction - gt_image)**2)
        self.assertLess(new_loss, loss,
                        f"New Loss ({new_loss}) >= Old Loss ({loss})")


if __name__ == '__main__':
    unittest.main()
