
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
import torch

class TargetPolicySmoothing(TensorDictModuleBase):
    '''
    Target policy smoothing for value function regularization.

    Generates noise and clips it to a specified range. Noise is added to the 
    target actions, then noisy actions are clipped to the valid action range.

    Deterministic policies may exploit small errors in the value function, leading 
    to high variance in the target policy. Adding noise to the actions and averaging 
    over mini-batches reduces this variance, thus regularizing the value function.

    **I/O**: *["next_act"] -> ["next_act"]*
    '''

    def __init__(
        self,
        noise_std: float = 0.2,
        noise_clip: float = 0.5,
        act_min: float = -1.0,
        act_max: float = 1.0,
    ):
        '''
        Initialize the target policy smoothing module.

        Parameters
        ----------
        noise_std : float, optional
            Standard deviation of the noise to be added to the target actions (default is 0.2).
        noise_clip : float, optional
            Clipping range for the noise (default is 0.5).
        act_min : float, optional
            Minimum value for the actions (default is -1.0).
        act_max : float, optional
            Maximum value for the actions (default is 1.0).
        '''
        super().__init__()
        self.in_keys = ["next_act"]
        self.out_keys = ["next_act"]

        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.act_min = act_min
        self.act_max = act_max

    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Apply target policy smoothing to the actions in the experience.

        Parameters
        ----------
        experience : TensorDict
            With existing keys:
            - "next_act": Actions from the target policy for the next state.

        Returns
        -------
        TensorDict
            With modified key:
            - "next_act": Actions with added noise.
        '''
        noise = torch.randn_like(experience["next_act"]) * self.noise_std
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        experience["next_act"] = torch.clamp(
            experience["next_act"] + noise, 
            self.act_min, 
            self.act_max
        )
        return experience