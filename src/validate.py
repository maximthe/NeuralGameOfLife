from train import run_training
from data import LifeDataset, generate_random_grid, game_of_life_step, K
import torch
import numpy as np


def grade_performance_single_step(model, grid_size=20, reverse=False):
    test = LifeDataset(grid_size, reverse=reverse)

    correct = 0

    model.eval()
    with torch.no_grad():
        for start, end in test:
            if (torch.round(model(start)) == torch.round(end)).all():
                correct += 1

    return correct / len(test)


def grade_performance_sequence(model, grid_size=20, num_steps=20):
    current_numpy = generate_random_grid(grid_size)

    model.eval()
    with torch.no_grad():
        for step in range(num_steps):
            # We use [0][0] later to get back to 2D, so we ensure 4D (Batch, Channel, H, W)
            input_tensor = torch.from_numpy(current_numpy).float().unsqueeze(0).unsqueeze(0)

            # Get Model Prediction
            pred_tensor = model(input_tensor)
            pred_numpy = torch.round(pred_tensor).squeeze().numpy()

            # Get Ground Truth
            actual_numpy = game_of_life_step(current_numpy, K)

            # Compare
            if np.array_equal(pred_numpy, actual_numpy):
                print(f"Step {step}: Correct")
                current_numpy = pred_numpy
            else:
                print(f"Error at step {step}")
                return False

    return True


if __name__ == "__main__":
    trained_model = run_training(grid_size=10, num_epochs=10, reverse=True)[0]
    correct = grade_performance_single_step(trained_model, grid_size=10, reverse=True)
    print(correct)
    correct = grade_performance_sequence(trained_model, num_steps=100, grid_size=500)
    print(correct)
