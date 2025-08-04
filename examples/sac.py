
from copy import deepcopy
from pymetheus.buffer import Buffer
from pymetheus.environment import Gymnasium
from pymetheus.policy import Policy, MLPPolicy
from pymetheus.policy.wrapper import TanhNormalPolicy
from pymetheus.modules import Entropy
from pymetheus.value import Value, MLPValue
from pymetheus.value.wrapper import EnsembleValue
from pymetheus.estimate import Estimate, TD0
from pymetheus.utils import disable_grads, enable_grads, polyak_update, prefix_keys
from tensordict import TensorDict
import torch

class SAC(torch.nn.Module):
    def __init__(
        self,
        policy_fn: Policy,
        value_fn: Value,
        entropy_reg: Entropy,
        estimate_fn: Estimate,
        distance_fn: torch.nn.Module
    ):
        super(SAC, self).__init__()

        # Policy and value functions
        self.policy_fn = TanhNormalPolicy(policy_fn)            # TanhNormal policy distribution
        self.value_fn = EnsembleValue(value_fn, num_copies=2)   # Clipped double Q-learning

        # Create targets as deep copies
        target_policy_fn = deepcopy(policy_fn)
        target_value_fn = deepcopy(value_fn)

        # Change the input and output keys of the target networks
        prefix_keys(target_policy_fn, "next_")
        prefix_keys(target_value_fn, "next_")

        # Target policy and value functions
        self.target_policy_fn = TanhNormalPolicy(target_policy_fn)
        self.target_value_fn = EnsembleValue(target_value_fn, num_copies=2)

        # Disable gradients for the target networks
        disable_grads(self.target_policy_fn, self.target_value_fn)

        # Functions used in value updates
        self.entropy_reg = entropy_reg
        self.estimate_fn = estimate_fn
        self.distance_fn = distance_fn

    def value_loss(self, batch: TensorDict) -> TensorDict:
        '''
        Compute the value loss for the current batch.

        Parameters
        ----------
        batch : TensorDict
            With existing keys:
            - "obs": Current observation.
            - "act": Action taken in the current observation.
            - "rew": Reward received for the action.
            - "next_obs": Observation after taking the action.
            - "done": Boolean indicating if the episode has ended.

        Returns
        -------
        batch : TensorDict
            With new or modified keys:
            - "next_act": The action selected by the target policy for the next observation.
            - "next_val": The next value estimate after applying the target policy and value functions.
            - "val": The Q-value for the current observation and action.
            - "val_loss": The computed value loss.
        '''
        with torch.no_grad():
            # Select action from the target policy given the next observation
            # ["next_obs"] -> ["next_act"]
            batch = self.target_policy_fn(batch)

            # Compute the target Q-value
            # ["next_obs", "next_act"] -> ["next_val"]
            batch = self.target_value_fn(batch)

            # Apply entropy regularization to target Q-value
            # ["next_val", "log_prob"] -> ["next_val"]
            batch = self.entropy_reg(batch)

            # Compute next-value estimate (Bellman backup)
            # ["next_val", "rew", "done"] -> ["next_val"]
            batch = self.estimate_fn(batch)

        # Compute clipped Q-value for the current observation and action
        # ["obs", "act"] -> ["val"]
        batch = self.value_fn(batch)
            
        # Compute the distance between the current Q-value and backup
        # ["val", "next_val"] -> ["val_loss"]
        batch["val_loss"] = torch.mean(
            self.distance_fn(batch["val"], batch["next_val"])
        )
        return batch

    def policy_loss(self, batch: TensorDict) -> TensorDict:
        '''
        Compute the policy loss for the current batch.

        Parameters
        ----------
        batch : TensorDict
            With existing keys:
            - "obs": Current observation.

        Returns
        -------
        batch : TensorDict
            With new or modified keys:
            - "act": The action selected by the policy for the current observation.
            - "log_prob": Log probability of the action taken.
            - "val": The Q-value for the current observation and action.
            - "pol_loss": The computed policy loss.
        '''

        # Select action given the current observation and policy
        # ["obs"] -> ["act", "log_prob"]
        batch = self.policy_fn(batch)
        
        # Compute the Q-value for the current observation and action
        # ["obs", "act"] -> ["val"]
        batch = self.value_fn(batch)
        
        # Compute the entropy-regularized policy loss
        # ["log_prob", "val"] -> ["pol_loss"]
        batch["pol_loss"] = torch.mean(
            self.entropy_reg.alpha * batch["log_prob"] - batch["val"]
        )
        return batch


    def entropy_loss(self, batch: TensorDict) -> TensorDict:
        '''
        Compute the entropy loss for the current batch.

        Parameters
        ----------
        batch : TensorDict
            With existing keys:
            - "log_prob": Log probability of the action taken.

        Returns
        -------
        batch : TensorDict
            With new or modified keys:
            - "ent_loss": The computed entropy loss.
        '''
        # Compute the entropy loss
        # ["log_prob"] -> ["ent_loss"]
        batch["ent_loss"] = -torch.mean(
            self.entropy_reg.log_alpha * \
            (batch["log_prob"] + self.entropy_reg.target_entropy).detach()
        )
        return batch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    environment = Gymnasium(id="Humanoid-v5", render_mode="human").to(device)
    proto_experience = environment.proto_experience

    # Initialize the agent
    tau = 0.005     # Polyak update coefficient
    target_entropy = Entropy.target_entropy_heuristic(proto_experience)
    agent = SAC(
        policy_fn=MLPPolicy(proto_experience, hidden_sizes=[1024, 1024, 1024]),
        value_fn=MLPValue(proto_experience, hidden_sizes=[1024, 1024, 1024]),
        entropy_reg=Entropy(target_entropy),
        estimate_fn=TD0(),
        distance_fn=torch.nn.MSELoss()
    ).to(device)

    # Optimizers
    policy_opt = torch.optim.Adam(agent.policy_fn.parameters(), lr=1e-3)
    value_opt = torch.optim.Adam(agent.value_fn.parameters(), lr=1e-3)
    alpha_opt = torch.optim.Adam(agent.entropy_reg.parameters(), lr=1e-3)

    # Initialize the experience replay buffer
    capacity = Buffer.calc_capacity(proto_experience, megabytes=16000)
    buffer = Buffer(proto_experience, capacity, batch_size=32)

    # Initialize experience as a copy of the prototype experience
    experience = environment.proto_experience.clone().to(device)

    # Initialize the exploration policy
    # explore_policy = EpsilonGreedyPolicy(agent.policy_fn, epsilon=0.1).to(device)
    
    # Main training loop
    while experience["episode"] < 10000:

        #################################
        ###  Environment Interaction  ###
        #################################

        with torch.no_grad():
            # Move to the next step in the environment
            # ["next_obs"] -> ["obs"]
            experience["obs"] = experience["next_obs"]

            # Select action given the current observation and policy
            # ["obs"] -> ["act"]
            experience = agent.policy_fn(experience)

            # Get initial state or feedback for taking the action
            # if ["done"]:  Clear experience, [...] -> ["obs"]
            # else:         ["obs", "act"] -> ["rew", "next_obs", "done"]
            experience = environment(experience)

            # Store complete experience in the buffer
            # Buffer will automatically discard the initial environment state
            buffer.add(experience)

        ######################
        ###  Agent Update  ###
        ######################

        if len(buffer) < buffer._batch_size: continue

        # Sample a batch of experiences from the buffer
        batch = buffer.sample().to(device)

        # Update value function
        value_opt.zero_grad()
        batch = agent.value_loss(batch)
        batch["val_loss"].backward()
        value_opt.step()

        # Disable value function gradients during policy update
        disable_grads(agent.value_fn)

        # Update policy
        policy_opt.zero_grad()
        batch = agent.policy_loss(batch)
        batch["pol_loss"].backward()
        policy_opt.step()

        # Re-enable value function gradients
        enable_grads(agent.value_fn)

        # Update entropy temperature
        alpha_opt.zero_grad()
        batch = agent.entropy_loss(batch)
        batch["ent_loss"].backward()
        alpha_opt.step()

        # Update target networks
        polyak_update(agent.target_value_fn, agent.value_fn, tau)
        polyak_update(agent.target_policy_fn, agent.policy_fn, tau)

        if experience["episode"] % 1 == 0:
            print(f"Episode: {experience['episode'].item():>4} | Step: {experience['step'].item():>4} | Rew: {experience['rew'].item():>16.6f} | Val Loss: {batch['val_loss'].item():>16.6f} | Pol Loss: {batch['pol_loss'].item():>16.6f}")
