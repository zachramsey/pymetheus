
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from abc import ABC, abstractmethod

class Reset(TensorDictModule, ABC):
    '''
    Abstract Base Class for resetting reinforcement learning environments.

    **Forward I/O**: [] -> ['state']
    '''

    in_keys = []
    out_keys = ['state']

    @abstractmethod
    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Reset the environment and return the initial state.
        
        Parameters
        ----------
        experience : TensorDict
            A TensorDict with the keys:
            - None

        Returns
        -------
        TensorDict
            A TensorDict with the new or modified keys:
            - 'state' : Tensor
                The initial state of the environment after reset.
        '''
        pass
