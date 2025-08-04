
from .. import Policy

from tensordict import TensorDict
from tensordict.nn import ProbabilisticTensorDictModule, NormalParamExtractor

from torch import nn, Tensor
import torch.distributions as D
from torch.distributions import constraints


class TanhNormal(D.TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True
    base_dist = D.Normal

    def __init__(self, loc, scale, validate_args=None):
        base_dist = D.Normal(loc, scale, validate_args=validate_args)
        transform = D.TanhTransform(cache_size=1)
        super().__init__(base_dist, transform, validate_args=validate_args)

    @property
    def loc(self) -> Tensor:
        return self.base_dist.loc
    
    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale
    
    @property
    def mean(self) -> Tensor:
        return 


class TanhNormalPolicy(ProbabilisticTensorDictModule):
    '''
    
    '''
    def __init__(self, policy: Policy):
        '''
        Parameters
        ----------
        policy : Policy
            The policy to wrap.
        '''
        super(TanhNormalPolicy, self).__init__(
            in_keys=["loc", "scale"],
            out_keys=["act"],
            default_interaction_type="random",
            distribution_class=TanhNormal,
            return_log_prob=True,
            log_prob_key="log_prob",
        )
        self.policy = policy
        self.param_out = nn.Linear(self.policy.out_dim, 2 * self.policy.out_dim)
        self.param_extractor = NormalParamExtractor()

    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Forward pass through the policy.

        Parameters
        ----------
        experience : TensorDict
            With existing keys:
            - "obs": Current observation.
            
        Returns
        -------
        experience : TensorDict
            With new or modified keys:
            - "act": The action selected by the policy for the current observation.
            - "log_prob": Log probability of the action taken.
        '''
        experience = self.policy(experience)        # Pass the experience through the base policy
        params = self.param_out(experience["act"])  # Get parameters for the action distribution
        loc, scale = self.param_extractor(params)   # Extract location and scale parameters
        experience.set("loc", loc)                  # Add location parameter to experience
        experience.set("scale", scale)              # Add scale parameter to experience
        experience.exclude("act")                   # Exclude the original action key
        experience = super().forward(experience)    # Feed experience to the distribution
        return experience
