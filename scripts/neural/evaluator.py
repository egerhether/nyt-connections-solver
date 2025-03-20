import torch.nn as nn

class NeuralEvaluator(nn.Module):

    def __init__(self, num_ftrs: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_ftrs, 150),
            nn.ReLU(),
            nn.Linear(150, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x