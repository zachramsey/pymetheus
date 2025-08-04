
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from abc import ABC, abstractmethod

class Reward(TensorDictModuleBase, ABC):
    '''
    Abstract base class for reward functions.
    
    **Forward I/O**: ["state", "action", "next_state"] -> ["reward"]
    '''

    in_keys = ['state', 'action', 'next_state']
    out_keys = ['reward']

    @abstractmethod
    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Computes the reward for taking an action in a given state and transitioning to the next state.

        Parameters
        ----------
        experience : TensorDict
            A TensorDict with the keys:
            - 'state': The current state of the environment.
            - 'action': The action taken by the agent.
            - 'next_state': The next state of the environment after taking the action.

        Returns
        -------
        TensorDict
            A TensorDict with the new or modified keys:
            - 'reward': The computed reward for the action taken in the state.
        '''
        pass
