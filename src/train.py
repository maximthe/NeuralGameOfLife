import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data import LifeDataset
from src.model import GameOfLifeNet


def train(model, optimizer, criterion, num_epochs):
    loss_history = [np.inf]
    for epoch in range(num_epochs):
        print(F"Epoch {epoch}/{num_epochs - 1}")
        print(f"Min Loss: {np.min(loss_history)}")

        avg_loss = 0

        for input, target in dataloader:
            # Prediction step
            prediction = model(input)
            loss = criterion(prediction, target)
            avg_loss += loss.item()

            # Learning step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_history.append(avg_loss / len(dataloader))


if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 64  # A common size for small image tasks
    LR = 0.001  # Learning rate
    NUM_EPOCHS = 50

    # Prepare Data
    dataset = LifeDataset(num_samples=10000, grid_size=20)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize tools
    model = GameOfLifeNet()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # Train
    train(model, optimizer, criterion, NUM_EPOCHS)