
from .. import Value

from tensordict import TensorDict
from tensordict.nn import EnsembleModule
import torch

class EnsembleValue(EnsembleModule):
    '''
    Wrapper for ensemble value functions; e.g., for clipped double Q learning in TD3 and SAC.

    **Forward I/O**: ["state", "action"] -> ["val"]
    '''

    def __init__(self, value_fn: Value, num_copies: int = 2):
        '''
        Initialize the ensemble value function.

        Parameters
        ----------
        value_fn : Value
            The value function to be used.
        num_copies : int, optional
            The number of copies of the value function to create in the ensemble (default is 2).
        '''
        super().__init__(value_fn, num_copies)

    def parameters(self, recurse: bool = True):
        '''
        Get the parameters of all value function instances.

        Parameters
        ----------
        recurse : bool, optional
            Whether to recursively get parameters from submodules (default is True).

        Returns
        -------
        Iterable[torch.nn.Parameter]
            An iterable of parameters from the value function instances.
        '''
        return self.params_td.parameters(recurse=recurse)
    
    def q1(self, experience: TensorDict) -> TensorDict:
        '''
        Compute the first Q-value from the value function instances.

        Parameters
        ----------
        experience : TensorDict
            A TensorDict with the keys:
            - 'state': The current state of the environment.
            - 'action': The action taken by the agent.

        Returns
        -------
        TensorDict
            A TensorDict with the new or modified keys:
            - 'val': The computed Q-value from the first value function instance.
        '''
        with self.params_td[0].to_module(self.module):
            experience = self.module(experience)
        return experience

    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Forward pass to compute the clipped Q-value.

        Parameters
        ----------
        experience : TensorDict
            A TensorDict with the keys:
            - 'state': The current state of the environment.
            - 'action': The action taken by the agent.

        Returns
        -------
        TensorDict
            A TensorDict with the new or modified keys:
            - 'val': The clipped Q-value computed from the value function instances.
        '''
        experience[self.out_keys[0]] = torch.min(super().forward(experience)[self.out_keys[0]])
        return experience
