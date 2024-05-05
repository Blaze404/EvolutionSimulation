# import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class PreyNN2(nn.Module):
    def __init__(self):
        super(PreyNN2, self).__init__()
        # Input layer to hidden layer
        self.hidden = nn.Linear(29, 8)
        # Hidden layer to output layer
        self.output = nn.Linear(8, 2)

    def forward(self, x):
        # Pass the input through the hidden layer, then apply ReLU activation
        x = F.relu(self.hidden(x))
        # Pass through output layer
        x = self.output(x)
        # Apply sigmoid activation function to ensure output is between 0 and 1
        x = torch.sigmoid(x)
        return x


class PreyNN(nn.Module):
    def __init__(self):
        super(PreyNN, self).__init__()
        self.fc1 = nn.Linear(29, 16)  # First fully connected layer
        # self.fc2 = nn.Linear(32, 16)  # Second fully connected layer
        self.angle = nn.Linear(16, 1)  # Output layer for angle
        self.magnitude = nn.Linear(16, 1)  # Output layer for magnitude

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        angle = torch.sigmoid(self.angle(x)) * 360  # Scale angle to 0-360
        magnitude = torch.sigmoid(self.magnitude(x))  # Scale magnitude to 0-1
        return angle, magnitude


class PredatorNN(nn.Module):
    def __init__(self):
        super(PredatorNN, self).__init__()
        self.fc1 = nn.Linear(19, 8)  # First fully connected layer
        # self.fc2 = nn.Linear(32, 16)  # Second fully connected layer
        self.angle = nn.Linear(8, 1)  # Output layer for angle
        self.magnitude = nn.Linear(8, 1)  # Output layer for magnitude

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        angle = torch.sigmoid(self.angle(x)) * 360  # Scale angle to 0-360
        magnitude = torch.sigmoid(self.magnitude(x))  # Scale magnitude to 0-1
        return angle, magnitude



def get_normal_dist_random_number(mean, sigma):
    s = np.random.normal(0, 0.5, 1)
    return float(s[0])


def mutate_weights(model, mutation_rate=0.01, mutation_effect=0.5):
    with torch.no_grad():  # Ensure no gradients are computed during mutation
        # Step 1: Collect all weights into a single list and find total number of weights
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.flatten())

        # Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)

        for i in range(len(all_weights)):
            if np.random.choice([1, 0], p=[mutation_rate, 1 - mutation_rate]):
                noise = get_normal_dist_random_number(0, mutation_effect)
                all_weights[i] += noise

        # Step 4: Write the mutated weights back to the original model parameters
        start_index = 0
        for param in model.parameters():
            if param.requires_grad:
                end_index = start_index + param.numel()
                # Reshape flattened weights back to the original shape
                param.data.copy_(all_weights[start_index:end_index].reshape(param.size()))
                start_index = end_index
