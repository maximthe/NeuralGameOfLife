import torch
from torch.utils.data import Dataset
import scipy.signal as conv
import numpy as np

# Convolution matrix
K = np.array([[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

def generate_random_grid(size, density):
    """
    :param size: numpy array (N, N)
    :param density: percentage of the grid filled in
    :return: numpy array (N, N) with 0's and 1's
    """
    # Initialize the grid (I'll stick to square ones)
    num_cells = size * size
    num_ones = int(num_cells * density)

    # Randomly distribute the live cells
    arr = np.zeros(num_cells, dtype=int)
    arr[:num_ones] = 1
    np.random.shuffle(arr)

    # Convert into a grid
    out = arr.reshape((size, size))
    return out

def game_of_life_step(grid, kernel):
    """
    :param grid: numpy array (N, N)
    :return: numpy array (N, N) next state
    """
    # Uses convolutions to efficiently calculate neighbor counts for the whole grid
    neighbor_counts = conv.convolve2d(grid, kernel, mode='same', boundary='wrap')

    # Simplified cell update logic
    new_grid = (neighbor_counts == 3) + grid * (neighbor_counts == 2)

    return new_grid


class LifeDataset(Dataset):
    def __init__(self, num_samples=10000, grid_size=20, density=0.20, seed=None):
        """
        Generates a dataset of valid state transitions on-the-fly
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.density = density
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Set seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed + idx)

        current_state = generate_random_grid(self.grid_size, self.density)
        next_state = game_of_life_step(current_state, K)

        # Convert to tensors
        current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

        return current_state_tensor, next_state_tensor


if __name__ == "__main__":
    test = LifeDataset(grid_size=20)
    print(test.__getitem__(0))
    print(test.__getitem__(1))