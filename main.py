import argparse
import torch
from src.train import run_training
from src.validate import grade_performance_single_step


def main():
    # Setup arguments, can tweak settings from the command line
    parser = argparse.ArgumentParser(description="Train a Neural Game of Life model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--grid_size", type=int, default=20, help="Size of the grid")
    parser.add_argument("--reverse", action="store_true", help="Task: Reverse the game (predict past from future)")
    args = parser.parse_args()

    print(f"Starting training for {args.epochs} epochs (Reverse task: {args.reverse})...")

    # Run Training
    model, loss_history = run_training(
        num_epochs=args.epochs,
        grid_size=args.grid_size,
        reverse=args.reverse,
        verbose=True
    )

    # Validate
    print("\nValidating model performance...")
    accuracy = grade_performance_single_step(model, grid_size=args.grid_size, reverse=args.reverse)
    print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    save_path = "neural_gol_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()