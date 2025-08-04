
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from abc import ABC, abstractmethod

class Value(TensorDictModuleBase, ABC):
    '''
    Abstract base class for value functions.

    **Forward I/O**: ["obs", "act"] -> ["val"]
    '''

    def __init__(self, proto_experience: TensorDict):
        '''
        Initialize the value function with a prototype experience.

        Parameters
        ----------
        proto_experience : TensorDict
            A TensorDict with the keys:
            - "obs": The observation tensor, which defines the observation dimension.
            - "act": The action tensor, which defines the action dimension.
        '''
        super().__init__()
        self.proto_experience = proto_experience
        self.in_keys = ['obs', 'act']
        self.out_keys = ['val']

        # Extract input dimension from the prototype experience
        self.in_dim = proto_experience["obs"].shape[-1] + proto_experience["act"].shape[-1]

    @abstractmethod
    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Computes the value estimate for a given state or state-action pair.
        
        Parameters
        ----------
        experience : TensorDict
            A TensorDict with the keys:
            - 'obs': The current state of the environment.
            - 'act': The action taken by the agent (ignored for state-value functions).

        Returns
        -------
        TensorDict
            A TensorDict with the new or modified keys:
            - 'val': The computed value estimate for the state or state-action pair.
        '''
        pass
