
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from abc import ABC, abstractmethod

class Transition(TensorDictModule, ABC):
    '''
    Abstract Base Class for transition functions in reinforcement learning environments.

    **Forward I/O**: ["state", "action"] -> ["next_state"]
    '''

    in_keys=['state', 'action']
    out_keys=['next_state']

    @abstractmethod
    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Determines the next state of the environment given the current state and action.
        
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
            - 'next_state': The next state of the environment after taking the action.
        '''
        pass
