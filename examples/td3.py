
from copy import deepcopy
from pymetheus.buffer import SumTree as Buffer
from pymetheus.environment import Gymnasium
from pymetheus.policy import Policy, MLPPolicy
from pymetheus.policy.wrapper import EpsilonGreedyPolicy
from pymetheus.modules import TargetPolicySmoothing
from pymetheus.value import Value, MLPValue
from pymetheus.value.wrapper import EnsembleValue
from pymetheus.estimate import Estimate, TD0
from pymetheus.utils import (
    disable_grads, enable_grads, polyak_update, prefix_keys)
from tensordict import TensorDict   # type: ignore
import torch


class TD3(torch.nn.Module):
    def __init__(
        self,
        policy_fn: Policy,
        value_fn: Value,
        estimate_fn: Estimate
    ):
        super(TD3, self).__init__()

        # Policy function
        self.policy_fn = policy_fn
        # Clipped double Q-learning value function
        self.value_fn = EnsembleValue(value_fn, num_copies=2)

        # Create targets as deep copies
        target_policy_fn = deepcopy(policy_fn)
        target_value_fn = deepcopy(value_fn)

        # Change the input and output keys of the target networks
        prefix_keys(target_policy_fn, "next_")
        prefix_keys(target_value_fn, "next_")

        # Target policy and value functions
        self.target_policy_fn = target_policy_fn
        self.target_value_fn = EnsembleValue(target_value_fn, num_copies=2)

        # Disable gradients for the target networks
        disable_grads(self.target_policy_fn, self.target_value_fn)

        # Functions used in value updates
        self.smoothing_reg = TargetPolicySmoothing()
        self.estimate_fn = estimate_fn

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
            - "next_act": Action selected for the next observation.
            - "next_val": Next value estimate.
            - "val": Q-value for the current observation and action.
            - "val_loss": The computed value loss.
        '''
        with torch.no_grad():
            # Select action from the target policy given the next observation
            # ["next_obs"] -> ["next_act"]
            batch = self.target_policy_fn(batch)

            # Apply target policy smoothing regularization
            # ["next_act"] -> ["next_act"]
            batch = self.smoothing_reg(batch)

            # Compute the target Q-value
            # ["next_obs", "next_act"] -> ["next_val"]
            batch = self.target_value_fn(batch)

            # Compute next-value estimate (Bellman backup)
            # ["next_val", "rew", "done"] -> ["next_val"]
            batch = self.estimate_fn(batch)

        # Compute Q-value for the current observation and action
        # ["obs", "act"] -> ["val"]
        batch = self.value_fn(batch)

        # Compute the TD error
        # ["val", "next_val"] -> ["td_error"]
        batch.set("td_error", batch["val"] - batch["next_val"])

        # Compute the distance between the current Q-value and backup
        # ["td_error", "weight"] -> ["val_loss"]
        batch.set("val_loss", torch.mean(torch.square(batch["td_error"]) *
                                         batch.get("weight", 1.0)))

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
            - "act": Action selected for the current observation.
            - "val": Q-value for the current observation and action.
            - "pol_loss": The computed policy loss.
        '''
        # Select action given the current observation and policy
        # ["obs"] -> ["act"]
        batch = self.policy_fn(batch)

        # Compute the Q-value for the current observation and action
        # ["obs", "act"] -> ["val"]
        batch = self.value_fn.q1(batch)

        # Maximize Q-value by minimizing its negative
        # ["val"] -> ["pol_loss"]
        batch.set("pol_loss", -torch.mean(batch["val"]))

        return batch


if __name__ == "__main__":
    env_id = "Humanoid-v5"  # Gymnasium environment ID
    policy_lr = 1e-3        # Learning rate for the policy optimizer
    value_lr = 1e-3         # Learning rate for the value optimizer
    erb_capacity = 1000000  # Capacity of the experience replay buffer
    max_episodes = 10000    # Total number of training episodes
    policy_delay = 1        # Frequency of policy updates
    gamma = 0.99            # Discount factor
    tau = 0.005             # Polyak update coefficient

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    environment = Gymnasium(id=env_id, render_mode="human").to(device)
    proto_experience = environment.proto_experience

    # Initialize the agent
    hidden_layers = [1024, 1024]
    agent = TD3(
        policy_fn=MLPPolicy(proto_experience, hidden_layers=hidden_layers),
        value_fn=MLPValue(proto_experience, hidden_layers=hidden_layers),
        estimate_fn=TD0(gamma=gamma),
    ).to(device)

    # Optimizers
    policy_opt = torch.optim.Adam(agent.policy_fn.parameters(), lr=policy_lr)
    value_opt = torch.optim.Adam(agent.value_fn.parameters(), lr=value_lr)

    # Initialize the experience replay buffer
    buffer = Buffer(proto_experience,
                    erb_capacity,
                    batch_size=256,
                    priority_key="td_error")

    # Initialize experience as a copy of the prototype experience
    experience: TensorDict = proto_experience.clone().to(device)

    # Initialize the exploration policy
    collect_policy = EpsilonGreedyPolicy(agent.policy_fn, epsilon=0.1)
    collect_policy = collect_policy.to(device)

    # Main training loop
    while experience["episode"] < max_episodes:

        #############################
        #  Environment Interaction  #
        #############################

        with torch.no_grad():
            # Move to the next step in the environment
            # ["next_obs"] -> ["obs"]
            experience.set("obs", experience["next_obs"], inplace=True)

            # Select action given the current observation and policy
            # ["obs"] -> ["act"]
            experience = collect_policy(experience)

            # Get initial state or feedback for taking the action
            # if ["done"]:  Clear experience, [...] -> ["obs"]
            # else:         ["obs", "act"] -> ["rew", "next_obs", "done"]
            experience = environment(experience)

            # Store complete experience in the buffer
            # Buffer will automatically discard the initial environment state
            buffer.add(experience)

        ##################
        #  Agent Update  #
        ##################

        # Only update once the buffer has enough samples
        if len(buffer) < buffer._batch_size:
            continue

        # Sample a batch of experiences from the buffer
        batch, idcs, _ = buffer.sample()
        batch = batch.to(device)

        # Update value function
        value_opt.zero_grad()
        batch = agent.value_loss(batch)
        batch["val_loss"].backward()
        value_opt.step()

        # Put experience back into the buffer
        buffer.update(idcs, batch)

        # Delayed policy update
        if experience["step"] % policy_delay == 0:

            # Disable value function gradients during policy update
            disable_grads(agent.value_fn)

            # Update policy
            policy_opt.zero_grad()
            batch = agent.policy_loss(batch)
            batch["pol_loss"].backward()
            policy_opt.step()

            # Re-enable value function gradients
            enable_grads(agent.value_fn)

            # Update target networks with polyak averaging
            polyak_update(agent.target_value_fn, agent.value_fn, tau)
            polyak_update(agent.target_policy_fn, agent.policy_fn, tau)

            if experience["episode"] % 1 == 0:
                print(f"Episode: {experience['episode'].item():>4} | "
                      f"Step: {experience['step'].item():>4} | "
                      f"Rew: {experience['rew'].item():>16.6f} | "
                      f"Val Loss: {batch['val_loss'].item():>16.6f} | "
                      f"Pol Loss: {batch['pol_loss'].item():>16.6f}")
