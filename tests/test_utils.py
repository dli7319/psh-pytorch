import unittest
from parameterized import parameterized
import torch

from psh_pytorch import utils


class TestUtils(unittest.TestCase):

    @parameterized.expand([
        [532, (10, 10, 10), (5, 3, 2)],
        [83, (10, 5, 2), (8, 1, 1)],
        [15, (5, 4), (3, 3)],
        [0, (5, 4), (0, 0)],
        [19, (30,), (19,)],
    ])
    def test_unravel_index(self, raveled_index, shape, expected_index):
        raveled_index = torch.tensor(raveled_index)
        expected_index = torch.tensor(expected_index)
        unraveled_index = utils.unravel_index(raveled_index, shape)
        self.assertTrue(torch.allclose(unraveled_index, expected_index),
                        f"Expected {expected_index}, got {unraveled_index}")

    @parameterized.expand([
        [(5, 3, 2), (10, 10, 10), 532],
        [(8, 1, 1), (10, 5, 2), 83],
        [(3, 3), (5, 4), 15],
        [(0, 0), (5, 4), 0],
        [19, (30,), (19,)],
    ])
    def test_ravel_index(self, index, shape, expected_raveled_index):
        index = torch.tensor(index)
        expected_raveled_index = torch.tensor(expected_raveled_index)
        raveled_index = utils.ravel_multi_index(index, shape)
        self.assertTrue(torch.allclose(raveled_index, expected_raveled_index),
                        f"Expected {expected_raveled_index}, got {raveled_index}")


if __name__ == '__main__':
    unittest.main()
