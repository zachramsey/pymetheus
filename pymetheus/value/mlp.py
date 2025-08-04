
from . import Value

import torch
from torch import nn

class MLPValue(Value):
    '''
    A basic multi-layer perceptron (MLP) value function that outputs action 
    values based on a non-linear transformation of the input observations and actions.

    *["obs", "act"] -> ["val"]*
    '''
    def __init__(self, proto_experience, hidden_layers: list=[64, 64], activation: nn.Module=torch.nn.ReLU):
        '''
        Initialize the MLP value function.
        
        Parameters
        ----------
        proto_experience : TensorDict
            With existing items:
            - "obs": The observation tensor, which defines the observation dimension.
            - "act": The action tensor, which defines the action dimension.
        '''
        super().__init__(proto_experience)

        # Define the layers for the MLP value function
        self.mlp = nn.Sequential()
        in_dim = self.in_dim
        for i, hidden_dim in enumerate(hidden_layers):
            self.mlp.add_module(f"fc{i}", nn.Linear(in_dim, hidden_dim))
            self.mlp.add_module(f"act{i}", activation())
            in_dim = hidden_dim
        self.mlp.add_module("output", nn.Linear(in_dim, 1))  # Output a single value

    def forward(self, experience):
        # Apply the MLP transformation to the observations and actions
        experience[self.out_keys[0]] = self.mlp(
            torch.cat([experience[self.in_keys[0]], experience[self.in_keys[1]]], dim=-1)
        )
        
        return experience