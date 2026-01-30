import torch.nn as nn


class GameOfLifeNet(nn.Module):
    def __init__(self):
        super(GameOfLifeNet, self).__init__()

        # Layer 1: Feature Extraction (The Moore Neighborhood)
        # We use padding_mode='circular' to handle the toroidal wrapping of the grid edges.
        # 1 input channel (the grid state) -> 16 hidden features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, padding_mode='circular')
        self.relu1 = nn.ReLU()

        # Layer 2: Logic Processing (Pixel-wise dense layer)
        # 1x1 convolution acts on each cell independently, analyzing the features from Layer 1.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()

        # Layer 3: Prediction (Output Probability)
        # Collapses the features back down to a single channel (alive or dead).
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input through the layers in sequence
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x