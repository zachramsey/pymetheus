
from . import Policy

import torch
from torch import nn

class MLPPolicy(Policy):
    '''
    A basic multi-layer perceptron (MLP) policy that outputs actions 
    based on a non-linear transformation of the input observations.

    *["obs"] -> ["act"]*
    '''
    def __init__(self, proto_experience, hidden_layers: list=[64, 64], activation: nn.Module=torch.nn.ReLU):
        '''
        Initialize the MLP policy.
        
        Parameters
        ----------
        proto_experience : TensorDict
            With existing items:
            - "obs": The observation tensor, which defines the observation dimension.
            - "act": The action tensor, which defines the action dimension.
        '''
        super().__init__(proto_experience)

        # Define the layers for the MLP policy
        self.mlp = nn.Sequential()
        in_dim = self.in_dim
        for i, hidden_dim in enumerate(hidden_layers):
            self.mlp.add_module(f"fc{i}", nn.Linear(in_dim, hidden_dim))
            self.mlp.add_module(f"act{i}", activation())
            in_dim = hidden_dim
        self.mlp.add_module("output", nn.Linear(in_dim, self.out_dim))

    def forward(self, experience):
        # Apply the MLP transformation to the observations
        experience[self.out_keys[0]] = self.mlp(experience[self.in_keys[0]])
        
        return experience
