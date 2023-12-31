from typing import Optional, Callable

import numpy as np
import torch
from torch import nn
import tqdm

from . import utils


class PerfectSpatialHash(nn.Module):
    def __init__(self, occupancy_grid: Optional[torch.Tensor], feature_channels: int,
                 fast_construction: bool = True,
                 fast_construction_offset_table_progression: float = 1.1,
                 dimensions: Optional[int] = None,
                 hash_table_size: Optional[int] = None,
                 offset_table_size: Optional[int] = None,
                 build_offset_table: bool = True,
                 verbose: bool = False):
        """Initialize the perfect hash.

        Args:
            occupancy_grid: The occupancy grid as D-dimensional tensor.
            feature_channels: The number of channels of the hash table.
            fast_construction: If true, avoids binary searching over the offset table size.
            fast_construction_offset_table_progression: (Optional) The progression factor of the offset table size.
            build_offset_table: (Optional) Whether to compute the offset table. Requires the occupancy grid.
            dimensions: (Optional) The number of dimensions of the hash table. Infered from the occupancy grid
              if available.
            hash_table_size: (Optional) The size of the hash table to initialize. Infered from the occupancy
              grid if available.
            offset_table_size: (Optional) The initial size of the offset table to test. Infered from the occupancy
              grid if unspecified.
            verbose: (Optional) Whether to print additional information.
        """
        super().__init__()
        if occupancy_grid is not None:
            self.dimensions = len(occupancy_grid.shape)
            num_values = torch.sum(occupancy_grid).item()
            device = occupancy_grid.device
            occupancy_grid = occupancy_grid.bool()
        else:
            assert dimensions is not None, "Either occupancy_grid or dimensions must be specified."
            assert hash_table_size is not None, "Either occupancy_grid or hash_table_size must be specified."
            assert offset_table_size is not None, "Either occupancy_grid or offset_table_size must be specified."
            assert not build_offset_table, "Cannot build offset table without occupancy grid."
            self.dimensions = dimensions
            num_values = 1
            device = torch.device("cpu")
        self.verbose = verbose
        self.verbose and print("Number of values:", num_values)

        self.hash_table_size = hash_table_size or int(np.ceil(
            num_values ** (1 / self.dimensions)))
        self.verbose and print("Hash table size:", self.hash_table_size)
        if self.hash_table_size > 256:
            self.hash_table_size = int(np.ceil(
                (1.01 * num_values) ** (1 / self.dimensions)))
        self.hash_table = nn.Parameter(torch.zeros(
            (self.hash_table_size,) * self.dimensions + (feature_channels,),
            dtype=torch.float32, device=device))

        sigma = 1 / (2 * self.dimensions)
        initial_offset_table_size = offset_table_size or int(np.ceil(
            (sigma * num_values) ** (1 / self.dimensions)))
        self.offset_table_size = initial_offset_table_size
        self.verbose and print("Offset table size:", self.offset_table_size)
        self.offset_table = nn.Parameter(torch.zeros(
            (self.offset_table_size,) * self.dimensions + (self.dimensions,),
            dtype=torch.uint8, device=device), requires_grad=False)

        self.m0 = torch.ones((self.dimensions,),
                             dtype=torch.float32, device=device)
        self.m1 = torch.ones((self.dimensions,),
                             dtype=torch.float32, device=device)
        self.oscale = np.ceil(self.hash_table_size / 255)

        while build_offset_table and not self._build_offset_table(occupancy_grid):
            # Increase the offset table size.
            new_offset_table_size = int(max(
                np.round(fast_construction_offset_table_progression *
                         self.offset_table_size),
                self.offset_table_size + 1))
            self.verbose and print(
                f"Increasing offset table size to {new_offset_table_size}")
            self.offset_table_size = new_offset_table_size
            self.offset_table = nn.Parameter(torch.zeros(
                (self.offset_table_size,) *
                self.dimensions + (self.dimensions,),
                dtype=torch.uint8, device=device), requires_grad=False)
        # Reset the hash table.
        self.hash_table.data = torch.rand_like(self.hash_table.data)
        if build_offset_table and not fast_construction:
            # TODO: Binary search for the offset table size.
            # low = initial_offset_table_size
            # high = self.offset_table_size + 1
            raise NotImplementedError("Slow construction not implemented yet.")

        if build_offset_table:
            self.sparsity_encoding = self._build_sparsity_encoding(
                occupancy_grid)
        else:
            self.sparsity_encoding = nn.Parameter(torch.zeros(
                tuple(self.hash_table.shape[:-1]) + (2,),
                dtype=torch.uint8, device=device),
                requires_grad=False)

        self._register_load_state_dict_pre_hook(
            self._update_parameter_sizes_from_state_dict)

    def _build_offset_table(self, occupancy_grid: torch.Tensor) -> bool:
        """Build the offset table.

        Returns:
            bool: True if the offset table was built successfully.
        """
        # Zero the hash table
        self.hash_table.data.zero_()
        occupied_indices = occupancy_grid.nonzero(as_tuple=False)
        occupancy_grid_size = occupancy_grid.shape[0]
        occupied_indices_ohashed = self._compute_offset_hash(occupied_indices)
        occupied_indices_ohashed_raveled = utils.ravel_multi_index(
            occupied_indices_ohashed, self.offset_table.shape[:-1])
        occupied_indices_bincount = torch.bincount(
            occupied_indices_ohashed_raveled, minlength=self.offset_table_size ** self.dimensions)
        # Order to fill the offset table.
        ordering = torch.argsort(occupied_indices_bincount, descending=True)
        offset_table_indices_sorted = utils.unravel_index(
            torch.arange(self.offset_table_size ** self.dimensions,
                         device=self.offset_table.device)[ordering],
            self.offset_table.shape[:-1])

        # Check that points for each offset position do not collide.
        for current_index in tqdm.tqdm(range(ordering.shape[0]), desc="Checking offset table collisions",
                                       disable=not self.verbose):
            if occupied_indices_bincount[ordering[current_index]] == 0:
                continue
            pixels = occupied_indices_ohashed_raveled == ordering[current_index]
            indices = occupied_indices[pixels]
            indices_h0 = (indices * self.m0).long() % self.hash_table_size
            indices_h0 = utils.ravel_multi_index(
                indices_h0, self.hash_table.shape[:-1])
            if (torch.bincount(indices_h0) > 1).any():
                return False

        offset_table_assigned = torch.zeros_like(self.offset_table[..., 0])

        current_index = 0
        # Assign the first offset.
        self.offset_table[offset_table_indices_sorted[current_index]] = torch.randint(
            0, 256, (self.dimensions,), dtype=torch.uint8, device=self.offset_table.device)
        used_pixels = occupied_indices_ohashed_raveled == ordering[current_index]
        self.hash_table.data[self.compute_hash(
            occupied_indices[used_pixels]).unbind(-1)] = 1
        offset_table_assigned[offset_table_indices_sorted[current_index].unbind(
            -1)] = 1

        for current_index in tqdm.tqdm(range(1, ordering.shape[0]), desc="Building offset table",
                                       disable=not self.verbose):
            tested_offsets = set()
            pixels = occupied_indices_ohashed_raveled == ordering[current_index]
            pixel_indices = occupied_indices[pixels]
            pixel_indices_hashed = (pixel_indices * self.m0).long()
            current_offset_index = offset_table_indices_sorted[current_index]
            offset_assigned = False

            def assign_offset(offset):
                """Check if this offset is valid."""
                nonlocal offset_assigned
                if offset_assigned:
                    return True
                if offset in tested_offsets:
                    return False
                tested_offsets.add(offset)
                hash_index = (pixel_indices_hashed +
                              offset) % self.hash_table_size
                values = self.hash_table[hash_index.unbind(-1)]
                if torch.any(values):
                    return False
                self.hash_table.data[hash_index.unbind(-1)] = 1
                self.offset_table[current_offset_index.unbind(-1)] = offset
                offset_table_assigned[current_offset_index.unbind(-1)] = 1
                offset_assigned = True
                return True

            # Heuristic 1: Try the neighboring pixels.
            offset_assigned or self._assign_offset_from_neighboring_pixels(
                pixel_indices, offset_table_assigned,
                occupancy_grid_size, assign_offset)

            # Heuristic 2: Try the neighboring offsets.
            offset_assigned or self._assign_offset_from_neighboring_offsets(
                current_offset_index, offset_table_assigned, assign_offset)

            # Heuristic 3: Try random offsets.
            offset_assigned or self._assign_offset_from_random_offsets(
                assign_offset)

            # Heuristic 4: Find empty positions in the hash table.
            offset_assigned or self._assign_offset_from_random_pixels(
                pixel_indices_hashed, assign_offset)

            if not offset_assigned:
                self.verbose and print(f"Could not assign offset with index {current_index} and " +
                                       f"pixels {occupied_indices_bincount[ordering[current_index]].item()}]")
                return False
        return True

    def _assign_offset_from_neighboring_pixels(self, pixel_indices: torch.Tensor,
                                               offset_table_assigned: torch.Tensor,
                                               occupancy_grid_size: int,
                                               assign_offset: Callable[[torch.Tensor], bool]) -> bool:
        """Heuristic: Assign an offset to the current pixel from neighboring pixels."""
        if len(pixel_indices) >= 10:
            return False
        for pixel_index in pixel_indices:
            for direction in range(self.dimensions):
                direction_offset = torch.zeros(
                    (self.dimensions,), dtype=torch.long, device=self.offset_table.device)
                direction_offset[direction] = 1
                index = (pixel_index +
                            direction_offset).clamp(0, occupancy_grid_size - 1)
                oindex = (
                    index * self.m1).long() % self.offset_table_size
                if (offset_table_assigned[oindex.unbind(-1)] and
                        assign_offset(self.offset_table[oindex.unbind(-1)].clone())):
                    return True
                direction_offset[direction] = -1
                index = (pixel_index +
                            direction_offset).clamp(0, occupancy_grid_size - 1)
                oindex = (
                    index * self.m1).long() % self.offset_table_size
                if (offset_table_assigned[oindex.unbind(-1)] and
                        assign_offset(self.offset_table[oindex.unbind(-1)].clone())):
                    return True
        return False

    def _assign_offset_from_neighboring_offsets(self,
                                                current_offset_index: torch.Tensor,
                                                offset_table_assigned: torch.Tensor,
                                                assign_offset: Callable[[torch.Tensor], bool]) -> bool:
        """Heuristic: Assign an offset to the current pixel from neighboring offsets."""
        for direction in range(self.dimensions):
            direction_offset = torch.zeros(
                (self.dimensions,), dtype=torch.long, device=self.offset_table.device)
            direction_offset[direction] = 1
            index = (current_offset_index +
                     direction_offset).clamp(0, self.offset_table_size - 1)
            if (offset_table_assigned[index.unbind(-1)] and
                    assign_offset(self.offset_table[index.unbind(-1)].clone())):
                return True
            direction_offset[direction] = -1
            index = (current_offset_index +
                     direction_offset).clamp(0, self.offset_table_size - 1)
            if (offset_table_assigned[index.unbind(-1)] and
                    assign_offset(self.offset_table[index.unbind(-1)].clone())):
                return True
        return False

    def _assign_offset_from_random_offsets(self, assign_offset: Callable[[torch.Tensor], bool]) -> bool:
        for _ in range(100):
            random_offset = torch.randint(
                0, 256, (self.dimensions,), dtype=torch.uint8, device=self.offset_table.device)
            if assign_offset(random_offset):
                return True
        return False

    def _assign_offset_from_random_pixels(self, pixel_indices_hashed: torch.Tensor,
                                          assign_offset: Callable[[torch.Tensor], bool]) -> bool:
        for _ in range(100):
            random_pixel = torch.randint(
                0, 256, (self.dimensions,), dtype=torch.uint8, device=self.offset_table.device)
            # Hashed position of the pixel.
            random_pixel_hashed = (
                random_pixel * self.m0).long() % self.hash_table_size
            # Check if the position is empty.
            if self.hash_table[random_pixel_hashed.unbind(-1)][0].item() == 0:
                offset = (random_pixel_hashed -
                          pixel_indices_hashed[0] + 10 * 255) % 256
                if assign_offset(offset):
                    return True
        return False

    def _build_sparsity_encoding(self, occupancy_grid: torch.Tensor):
        sparsity_k = torch.zeros(
            self.hash_table[..., 0].shape, dtype=torch.uint8, device=self.hash_table.device)
        sparsity_hk = torch.zeros(
            self.hash_table[..., 0].shape, dtype=torch.uint8, device=self.hash_table.device)

        occupied_indices = occupancy_grid.nonzero(as_tuple=False)
        occupied_indices_hashed = self.compute_hash(occupied_indices)
        sparsity_k[occupied_indices_hashed.unbind(-1)] = 1
        sparsity_hk[occupied_indices_hashed.unbind(
            -1)] = utils.sparsity_hash(1, occupied_indices)

        unoccupied_grid = ~occupancy_grid
        unoccupied_indices = unoccupied_grid.nonzero(as_tuple=False)
        unoccupied_indices_hashed = self.compute_hash(unoccupied_indices)
        for i in range(1, 256):
            unoccupied_indices_shashed = utils.sparsity_hash(
                sparsity_k[unoccupied_indices_hashed.unbind(-1)], unoccupied_indices)
            collisions = torch.logical_and(
                sparsity_k[unoccupied_indices_hashed.unbind(-1)] != 0,
                sparsity_hk[unoccupied_indices_hashed.unbind(-1)] == unoccupied_indices_shashed)
            number_of_collisions = collisions.sum()
            self.verbose and print(
                f"Number of collisions for {i} is {number_of_collisions}")
            if number_of_collisions == 0:
                break
            sparsity_k[unoccupied_indices_hashed[collisions].unbind(-1)] = 0
            if i < 255:
                collided_indices = sparsity_k[occupied_indices_hashed.unbind(
                    -1)] == 0
                sparsity_k[occupied_indices_hashed[collided_indices]
                           .unbind(-1)] = i+1
                sparsity_hk[occupied_indices_hashed[collided_indices].unbind(
                    -1)] = utils.sparsity_hash(i+1, occupied_indices[collided_indices])
        return nn.Parameter(torch.stack([sparsity_k, sparsity_hk], dim=-1),
                            requires_grad=False)

    def _compute_offset_hash(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the hash of the input tensor into the offset table.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C).

        Returns:
            torch.Tensor: Hash indices of shape (N,).
        """
        return (x * self.m1).long() % self.offset_table_size

    def compute_hash(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the hash of the input tensor into the hash table.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C).

        Returns:
            torch.Tensor: Hash indices of shape (N,).
        """
        h0 = (x * self.m0).long()
        h1 = self._compute_offset_hash(x)
        offset = self.offset_table[h1.unbind(-1)] * self.oscale
        return (h0 + offset).long() % self.hash_table_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the perfect hash function.

        Args:
            Tuple[torch.Tensor, torch.Tensor]: Hashed values and sparsity.
        """
        hash_indices = self.compute_hash(x)
        values = self.hash_table[hash_indices.unbind(-1)]
        sparsity_k, sparsity_hk = self.sparsity_encoding[hash_indices.unbind(
            -1)].unbind(-1)
        correct_hk = torch.logical_and(
            sparsity_k > 0, utils.sparsity_hash(sparsity_k, x) == sparsity_hk)
        return values, correct_hk

    def encode(self, positions: torch.Tensor, values: torch.Tensor):
        """Encode the input values into the hash table.

        Args:
            positions (torch.Tensor): Positions of the input values.
            values (torch.Tensor): Values to encode.
        """
        hash_indices = self.compute_hash(positions)
        self.hash_table.data[hash_indices.unbind(-1)] = values

    def check_collisions(self, positions: torch.Tensor) -> bool:
        """Check if there are any collisions at the given positions.

        Args:
            positions (torch.Tensor): Positions to check.

        Returns:
            bool: True if there are collisions, False otherwise.
        """
        hash_indices = self.compute_hash(positions)
        return torch.unique(hash_indices, dim=0).shape[0] != hash_indices.shape[0]

    def _update_parameter_sizes_from_state_dict(self,
                                                state_dict, prefix,
                                                local_metadata, strict,
                                                missing_keys, unexpected_keys, error_msgs):
        print(list(state_dict.keys()))
        new_hash_table = state_dict[prefix + "hash_table"]
        new_sparsity_encoding = state_dict[prefix + "sparsity_encoding"]
        if self.hash_table_size != new_hash_table.shape[0]:
            self.hash_table_size = new_hash_table.shape[0]
            self.hash_table.data = torch.zeros(
                new_hash_table.shape,
                dtype=self.hash_table.dtype,
                device=self.hash_table.device)
            self.sparsity_encoding.data = torch.zeros(
                new_sparsity_encoding.shape,
                dtype=self.sparsity_encoding.dtype,
                device=self.sparsity_encoding.device)
        new_offset_table = state_dict[prefix + "offset_table"]
        if self.offset_table_size != new_offset_table.shape[0]:
            self.offset_table_size = new_offset_table.shape[0]
            self.offset_table.data = torch.zeros(
                new_offset_table.shape,
                dtype=self.offset_table.dtype,
                device=self.offset_table.device)
