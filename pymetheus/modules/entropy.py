
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
import torch

class Entropy(TensorDictModuleBase):
    '''
    Automatic entropy temperature (alpha) for soft-learning algorithms like Soft Actor-Critic (SAC).

    **Forward I/O**: ["next_val", "log_prob"] -> ["next_val"]

    Attributes
    ----------
    alpha : Tensor
        The current value of the entropy temperature.
        Exponentiates the `log_alpha` to ensure the temperature is always positive.

    Static Methods
    --------------
    target_entropy_heuristic(proto_experience: TensorDict) -> float
        Target entropy selection heuristic; computes the negative norm of the action space.
    '''
    def __init__(self, target_entropy: float, alpha: float = 1.0):
        '''
        Initialize the entropy regularization module.

        Parameters
        ----------
        target_entropy : float
            The target entropy value for the policy.  
            *A heuristic for determining the target entropy is
            available via `Entropy.target_entropy_heuristic`.*
        alpha : float, optional
            The initial value for the entropy temperature (default is 1.0).
        '''
        super().__init__()
        self.in_keys = ["log_prob"]
        self.out_keys = ["entropy_loss"]

        # Target entropy for the policy
        self._target_entropy = target_entropy

        # Learnable entropy temperature parameter
        self._log_alpha = torch.nn.Parameter(torch.log(torch.tensor(alpha, dtype=torch.float32)))
        # The temperature is stored in the space of real numbers for easier optimization.
        # When using the temperature, it is exponentiated to always get a positive value.

    @property
    def target_entropy(self):
        '''
        Get the target entropy for the policy.
        '''
        return self._target_entropy
    
    @property
    def log_alpha(self):
        '''
        Get the logarithm of the current entropy temperature.
        '''
        return self._log_alpha

    @property
    def alpha(self):
        '''
        Get the current value of the entropy temperature.
        '''
        return torch.exp(self._log_alpha)
    
    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Apply entropy regularization to the next value estimate.

        Parameters
        ----------
        experience : TensorDict
            With existing keys:
            - "next_val": The next value estimate.
            - "log_prob": The log probabilities of the actions taken.

        Returns
        -------
        experience : TensorDict
            With new or updated keys:
            - "next_val": The next value estimate adjusted by the entropy temperature.
        '''
        experience["next_val"] = (
            experience["next_val"] - 
            torch.exp(self._log_alpha) * experience["log_prob"]
        )
        return experience

    @staticmethod
    def target_entropy_heuristic(proto_experience: TensorDict) -> float:
        '''
        Heuristic to select the target entropy based on the action space.

        Parameters
        ----------
        proto_experience : TensorDict
            A prototype experience containing a prototype action.

        Returns
        -------
        target_entropy : float
            The computed target entropy value.
        '''
        # Negative norm of the action space (shape of the prototype action)
        act_space = proto_experience["act"]
        return -torch.prod(torch.tensor(act_space)).item()