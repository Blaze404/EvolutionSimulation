import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class PreyNN(nn.Module):
    def __init__(self):
        super(PreyNN, self).__init__()
        # Input layer to hidden layer
        self.hidden = nn.Linear(29, 10)
        # Hidden layer to output layer
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        # Pass the input through the hidden layer, then apply ReLU activation
        x = F.relu(self.hidden(x))
        # Pass through output layer
        x = self.output(x)
        # Apply sigmoid activation function to ensure output is between 0 and 1
        x = torch.sigmoid(x)
        return x


class PredatorNN(nn.Module):
    def __init__(self):
        super(PredatorNN, self).__init__()
        # Input layer to hidden layer
        self.hidden = nn.Linear(14, 10)
        # Hidden layer to output layer
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        # Pass the input through the hidden layer, then apply ReLU activation
        x = F.relu(self.hidden(x))
        # Pass through output layer
        x = self.output(x)
        # Apply sigmoid activation function to ensure output is between 0 and 1
        x = torch.sigmoid(x)
        return x


def mutate_weights(model, mutation_rate=0.01, mutation_effect=0.2):
    with torch.no_grad():  # Ensure no gradients are computed during mutation
        # Step 1: Collect all weights into a single list and find total number of weights
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.flatten())

        # Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)
        total_weights = all_weights.numel()

        # Step 2: Calculate the number of weights to mutate
        num_mutations = int(total_weights * mutation_rate)

        # Randomly choose indices of weights to mutate
        indices_to_mutate = np.random.choice(range(total_weights), num_mutations, replace=False)

        # Step 3: Mutate the selected weights
        # Example mutation: adding random noise scaled by mutation_effect
        noise = torch.randn(num_mutations) * mutation_effect
        all_weights[indices_to_mutate] += noise

        # Step 4: Write the mutated weights back to the original model parameters
        start_index = 0
        for param in model.parameters():
            if param.requires_grad:
                end_index = start_index + param.numel()
                # Reshape flattened weights back to the original shape
                param.data.copy_(all_weights[start_index:end_index].reshape(param.size()))
                start_index = end_index
