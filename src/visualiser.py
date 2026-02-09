import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.data import game_of_life_step, K
from src.train import run_training


def create_life_gif(grid, n_frames, fps, filename='game_of_life.gif'):
    """
    Generates a Game of Life simulation (ground truth only) and saves it as a GIF.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')

    def update(frame):
        nonlocal grid
        grid = game_of_life_step(grid, K)
        img.set_data(grid)
        return img,

    print(f"Generating {n_frames} frames for {filename}...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    ani.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved as {filename}")


def compare_model_vs_reality(model, grid, n_frames, fps, filename='comparison.gif'):
    """
    Generates a side-by-side GIF comparing the Ground Truth (math) vs Model Prediction.

    :param model: The trained PyTorch model.
    :param grid: Initial numpy array (N, N).
    :param n_frames: Number of frames to simulate.
    :param fps: Frames per second.
    :param filename: Output filename.
    """

    # Initialize both grids with the same start state
    gt_grid = grid.copy()
    pred_grid = grid.copy()

    # Setup the plot with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Ground Truth Panel
    ax1.set_title("Ground Truth (Math)")
    ax1.axis('off')
    img1 = ax1.imshow(gt_grid, interpolation='nearest', cmap='binary', vmin=0, vmax=1)

    # Prediction Panel
    ax2.set_title("Model Prediction (Neural Net)")
    ax2.axis('off')
    img2 = ax2.imshow(pred_grid, interpolation='nearest', cmap='binary', vmin=0, vmax=1)

    model.eval()  # Set model to evaluation mode

    def update(frame):
        nonlocal gt_grid, pred_grid

        # --- Left: Calculate Ground Truth ---
        gt_grid = game_of_life_step(gt_grid, K)

        # --- Right: Calculate Prediction ---
        # Prepare input: Convert numpy (H, W) -> Tensor (1, 1, H, W)
        input_tensor = torch.tensor(pred_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Convert output probability map back to binary grid
        # We round > 0.5 to get the crisp 0 or 1 state
        pred_grid = (output_tensor.squeeze().numpy() > 0.5).astype(int)

        # Update the images
        img1.set_data(gt_grid)
        img2.set_data(pred_grid)

        return img1, img2

    print(f"Generating comparison: {filename} ({n_frames} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    ani.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved comparison to {filename}")


if __name__ == '__main__':
    # Train a fresh model
    print("Training a fresh model for visualization...")
    # Using 30 epochs and 20x20 grid size as a balanced default
    model, _ = run_training(
        num_samples=10000,
        grid_size=20,
        num_epochs=30,
        verbose=True
    )

    # 2. Define a Test Scenario (The "Glider" pattern)
    # Note: Although trained on 20x20, the fully convolutional nature allows 
    # applying it to a 15x15 grid.
    grid_size = 15
    initial_grid = np.zeros((grid_size, grid_size), dtype=int)

    # Glider coordinates
    # . O .
    # . . O
    # O O O
    offset = 3
    glider_coords = [(0 + offset, 1), (1 + offset, 2), (2 + offset, 0), (2 + offset, 1), (2 + offset, 2)]

    for r, c in glider_coords:
        initial_grid[r, c] = 1

    # 3. Generate the Comparison GIF
    compare_model_vs_reality(
        model=model,
        grid=initial_grid,
        n_frames=60,
        fps=10,
        filename='model_vs_reality_glider.gif'
    )