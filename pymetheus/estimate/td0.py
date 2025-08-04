
from . import Estimate

from tensordict import TensorDict

class TD0(Estimate):
    '''
    TD(0) (one-step temporal difference) value estimator.

    Updates the next-state value estimate based on 
    the reward and done flag from the experience.

    *["rew", "done", "next_val"] -> ["next_val"]*
    '''

    def __init__(self, gamma: float = 0.99):
        '''
        Initialize the TD(0) estimator.

        Parameters
        ----------
        gamma : float, optional
            The discount factor for future rewards.
        '''
        super().__init__()
        self.gamma = gamma

    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Update the next-state value estimate using TD(0).

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
        experience["next_val"] = (
            experience["rew"] + 
            self.gamma * (1 - experience["done"]) * experience["next_val"]
        )
        return experience
    