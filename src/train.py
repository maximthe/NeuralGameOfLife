import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data import LifeDataset
from src.model import GameOfLifeNet


def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Forward pass
        predictions = model(input_batch)
        loss = criterion(predictions, target_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(model, dataloader, optimizer, criterion, num_epochs, device='cpu', verbose=True):
    """Train the model for multiple epochs."""
    loss_history = []

    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f} - Min Loss: {min(loss_history):.6f}")

    return loss_history


def create_model_and_optimizer(lr=0.001):
    """Initialize model and optimizer."""
    model = GameOfLifeNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def create_dataloader(num_samples=10000, grid_size=20, batch_size=64, shuffle=True, reverse=False):
    """Create dataset and dataloader."""
    dataset = LifeDataset(num_samples=num_samples, grid_size=grid_size, reverse=reverse)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def run_training(
        num_samples=10000,
        grid_size=20,
        batch_size=64,
        lr=0.001,
        num_epochs=50,
        device='cpu',
        verbose=True,
        reverse=False
):
    """
    Main training function that can be called from other modules.

    Args:
        num_samples: Number of training samples
        grid_size: Size of the Game of Life grid
        batch_size: Batch size for training
        lr: Learning rate
        num_epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
        reverse: Whether predicting the future or previous move

    Returns:
        model: Trained model
        loss_history: List of average losses per epoch
    """
    # Setup
    dataloader = create_dataloader(num_samples, grid_size, batch_size, reverse)
    model, optimizer = create_model_and_optimizer(lr)
    criterion = nn.BCELoss()

    # Move model to device
    model = model.to(device)

    # Train
    loss_history = train(model, dataloader, optimizer, criterion, num_epochs, device, verbose)

    return model, loss_history


if __name__ == "__main__":
    # Train with default parameters
    model, loss_history = run_training(
        num_samples=10000,
        grid_size=20,
        batch_size=64,
        lr=0.001,
        num_epochs=50,
        device='cpu',
        verbose=True
    )

    print(f"\nTraining complete. Final loss: {loss_history[-1]:.6f}")