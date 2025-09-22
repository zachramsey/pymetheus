
from copy import deepcopy
from pymetheus.buffer import SumTree as Buffer
from pymetheus.environment import Gymnasium
from pymetheus.policy import Policy, MLPPolicy
from pymetheus.policy.wrapper import EpsilonGreedyPolicy
from pymetheus.value import Value, MLPValue
from pymetheus.estimate import Estimate, TD0
from pymetheus.utils import (
    disable_grads, enable_grads, polyak_update, prefix_keys)
from tensordict import TensorDict   # type: ignore
import torch


class DDPG(torch.nn.Module):
    def __init__(
        self,
        policy_fn: Policy,
        value_fn: Value,
        estimate_fn: Estimate
    ):
        super(DDPG, self).__init__()

        # Policy and value functions
        self.policy_fn = policy_fn
        self.value_fn = value_fn

        # Create targets as deep copies
        self.target_policy_fn = deepcopy(self.policy_fn)
        self.target_value_fn = deepcopy(self.value_fn)

        # Change the input and output keys of the target networks
        prefix_keys(self.target_policy_fn, "next_")
        prefix_keys(self.target_value_fn, "next_")

        # Disable gradients for the target networks
        disable_grads(self.target_policy_fn, self.target_value_fn)

        # Functions used in value updates
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
            - "next_act": The action selected for the next observation.
            - "next_val": The next value estimate.
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

            # Compute next-value estimate (Bellman backup)
            # ["next_val", "rew", "done"] -> ["next_val"]
            batch = self.estimate_fn(batch)

        # Compute Q-value for the current observation and action
        # ["obs", "act"] -> ["val"]
        batch = self.value_fn(batch)

        # Compute the TD error
        # ["val","next_val"] -> ["td_error"]
        batch.set("td_error", batch["val"] - batch["next_val"])

        # Compute the distance between the current Q-value and backup
        # ["td_error", "weight"] -> ["val_loss"]
        batch.set("val_loss", torch.mean(torch.square(batch["td_error"])
                                         * batch.get("weight", 1.0)))

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
            - "act": The action selected for the current observation.
            - "val": The Q-value for the current observation and action.
            - "pol_loss": The computed policy loss.
        '''
        # Select action given the current observation and policy
        # ["obs"] -> ["act"]
        batch = self.policy_fn(batch)

        # Compute the Q-value for the current observation and action
        # ["obs", "act"] -> ["val"]
        batch = self.value_fn(batch)

        # Maximize Q-value by minimizing its negative
        # ["val"] -> ["pol_loss"]
        batch["pol_loss"] = -torch.mean(batch["val"])

        return batch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    environment = Gymnasium(id="Humanoid-v5", render_mode="human").to(device)
    proto_experience = environment.proto_experience

    # Initialize the DDPG agent
    tau = 0.005     # Polyak update coefficient
    hidden_layers = [1024, 1024, 1024]
    agent = DDPG(
        policy_fn=MLPPolicy(proto_experience, hidden_layers=hidden_layers),
        value_fn=MLPValue(proto_experience, hidden_layers=hidden_layers),
        estimate_fn=TD0()
    ).to(device)

    # Initialize optimizers for policy and value functions
    policy_opt = torch.optim.Adam(agent.policy_fn.parameters(), lr=1e-3)
    value_opt = torch.optim.Adam(agent.value_fn.parameters(), lr=1e-3)

    # Initialize the experience replay buffer
    capacity = Buffer.calc_capacity(proto_experience, megabytes=16000)
    buffer = Buffer(proto_experience, capacity, batch_size=16)

    # Initialize experience as a copy of the prototype experience
    experience = proto_experience.clone().to(device)

    explore_policy = EpsilonGreedyPolicy(agent.policy_fn, epsilon=0.1)
    explore_policy = explore_policy.to(device)

    # Main training loop
    while experience["episode"] < 1000:

        #############################
        #  Environment Interaction  #
        #############################

        with torch.no_grad():
            # Move to the next step in the environment
            # ["next_obs"] -> ["obs"]
            experience["obs"] = experience["next_obs"]

            # Select action given the current observation and policy
            # ["obs"] -> ["act"]
            experience = explore_policy(experience)

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

        # Update buffer priorities
        buffer.update(idcs, batch)

        # Disable value function gradients during policy update
        disable_grads(agent.value_fn)

        # Update policy
        policy_opt.zero_grad()
        batch = agent.policy_loss(batch)
        batch["pol_loss"].backward()
        policy_opt.step()

        # Re-enable value function gradients
        enable_grads(agent.value_fn)

        # Update target networks
        polyak_update(agent.target_value_fn, agent.value_fn, tau)
        polyak_update(agent.target_policy_fn, agent.policy_fn, tau)

        if experience["episode"] % 1 == 0:
            # print(f"\nEpisode: {experience['episode'].item()} | "
            #       f"Step: {experience['step'].item()}")
            # print(f"Obs: {[f'{x:>6.2f}'
            #                for x in experience['obs'].tolist()]}")
            # print(f"Act: {[f'{x:>6.2f}'
            #                for x in experience['act'].tolist()]}")
            # print(f"Rew: {experience['rew'].item():>10.6f} | "
            #       f"Done: {experience['done'].item()}")
            # print(f"Val Loss: {batch['val_loss'].item():>10.6f} | "
            #       f"Pol Loss: {batch['pol_loss'].item():>10.6f}")
            print(f"Episode: {experience['episode'].item():>4} | "
                  f"Step: {experience['step'].item():>4} | "
                  f"Rew: {experience['rew'].item():>16.6f} | "
                  f"Val Loss: {batch['val_loss'].item():>16.6f} | "
                  f"Pol Loss: {batch['pol_loss'].item():>16.6f}")

    # Close the environment
    environment.close()
