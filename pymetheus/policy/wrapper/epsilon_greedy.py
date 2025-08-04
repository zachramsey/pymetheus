
from .. import Policy
from tensordict import TensorDict
import torch

class EpsilonGreedyPolicy(Policy):
    '''
    Policy Modifier for Epsilon-Greedy Exploration.

    '''
    def __init__(self, policy: Policy, epsilon: float = 0.1):
        '''
        Initialize the Epsilon-Greedy Policy Modifier.

        Parameters
        ----------
        policy : Policy
            The base policy to be modified.
        proto_experience : TensorDict
            The prototype experience, defining the action space.
        epsilon : float, optional
            The probability of taking a random action (default is 0.1).
        '''
        super().__init__(policy.proto_experience)
        self.policy = policy
        self.act_min = policy.proto_experience["act_min"]
        self.act_max = policy.proto_experience["act_max"]
        self.epsilon = epsilon

    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Apply the Epsilon-Greedy policy to the given experience.

        Parameters
        ----------
        experience : TensorDict
            With existing keys:
            - "obs": The current observation.

        Returns
        -------
        experience : TensorDict
            With new or modified keys:
            - "act": The action selected by the Epsilon-Greedy policy.
        '''
        # Generate a random number to decide whether to explore or exploit
        if torch.rand(1).item() < self.epsilon:
            # Explore: select a random action within the action bounds
            experience["act"] = torch.rand_like(self.act_min) * (self.act_max - self.act_min) + self.act_min
        else:
            # Exploit: use the base policy to select an action
            experience = self.policy(experience)

        return experience
