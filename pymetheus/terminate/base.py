
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from abc import ABC, abstractmethod

class Terminate(TensorDictModule, ABC):
    '''
    Abstract Base Class for termination functions in reinforcement learning environments.

    **Forward I/O**: ["state", "action", "next_state"] -> ["done"]
    '''

    in_keys = ['state', 'action', 'next_state']
    out_keys = ['done']

    @abstractmethod
    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Determines whether the episode has reached a terminal
        state based on the current state, action, and next state.
        
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
            - 'done': A flag indicating whether the episode has reached
                      a terminal state (1 if done, 0 otherwise).
        '''
        pass
