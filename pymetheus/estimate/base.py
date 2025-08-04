
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from abc import ABC, abstractmethod

class Estimate(TensorDictModuleBase, ABC):
    '''
    Abstract base class for value estimators.

    Updates the next-state value estimate based on 
    the reward and done flag from the experience.

    *["rew", "done", "next_val"] -> ["next_val"]*
    '''

    in_keys = ["rew", "done", "next_val"]
    out_keys = ["next_val"]

    @abstractmethod
    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Update the next-state value estimate.
        
        Parameters
        ----------
        experience : TensorDict
            With existing keys:
            - 'rew': The reward received after taking an action.
            - 'done': Flag indicating if the episode has ended (1 if done, 0 otherwise).
            - 'next_val': The estimated value of the next state.

        Returns
        -------
        TensorDict
            With new or modified keys:
            - 'next_val': The updated estimated value of the next state.
        '''
        pass
