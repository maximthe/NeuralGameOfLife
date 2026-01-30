import data
from data import K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_life_gif(grid, n_frames, fps, filename='game_of_life.gif', size=100):
    """
    Generates a Game of Life simulation and saves it as a GIF.

    :param n_frames (int): Total number of frames to record.
    :param fps (int): Frames per second for the output GIF.
    :param filename (str): Name of the file to save (e.g., 'simulation.gif').
    :param size (int): The width/height of the grid.
    """

    # Setup the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')  # Hide axes
    # 'binary' cmap gives black and white (0=white, 1=black by default)
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')

    # Animation Loop
    def update(frame):
        nonlocal grid
        grid = data.game_of_life_step(grid, K)
        img.set_data(grid)
        return img,  # returns a tuple with 1 element

    # Generate and Save
    print(f"Generating {n_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames)

    # 'pillow' is a built-in writer that handles GIFs nicely
    ani.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved as {filename}")
    

if __name__ == '__main__':
    # Simple glider
    grid = np.zeros((15, 15), dtype=int)

    # Shape:
    # . O .
    # . . O
    # O 0 O

    # Coordinates (Row, Col)
    k = 5  # offset
    glider_coords = [(0 + k, 1), (1 + k, 2), (2 + k, 0), (2 + k, 1), (2 + k, 2)]

    for r, c in glider_coords:
        grid[r, c] = 1

    create_life_gif(grid, n_frames=100, fps=10, filename='game_of_life_glider.gif')

    # R-Pentomino
    grid = np.zeros((30, 30), dtype=int)

    # The R-Pentomino Pattern
    #   . O O
    #   O O .
    #   . O .
    r_pentomino_coords = [(6, 7), (6, 8),
                          (7, 6), (7, 7),
                          (8, 7)]

    for r, c in r_pentomino_coords:
        grid[r, c] = 1

    create_life_gif(grid, n_frames=100, fps=30, filename='game_of_life_pentomino.gif')

    # The Diehard Pattern (vanishes in 130 gens on infinite grid)
    grid = np.zeros((30, 30), dtype=int)

    # Shape:
    #       . . . . . . O .
    #       O O . . . . . .
    #       . O . . . O O O

    diehard_coords = [
        (15, 12), (15, 13),  # Left-bottom 'block'
        (16, 13),  # Middle connector
        (14, 18),  # Top lonely cell
        (16, 17), (16, 18), (16, 19)  # Right 'bar'
    ]

    for r, c in diehard_coords:
        grid[r, c] = 1

    create_life_gif(grid, n_frames=150, fps=30, filename='game_of_life_diehard.gif')


