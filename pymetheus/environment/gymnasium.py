
from . import Environment

import numpy as np
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch
import torch.nn.functional as F

try:
    import gymnasium
    from gymnasium.envs.registration import EnvSpec
    from gymnasium import spaces
    from gymnasium.spaces.utils import flatten_space, flatten, unflatten
except ImportError:
    raise ImportError("Gymnasium could not be imported. Ensure it is installed; e.g., via 'pip install gymnasium'.")

class Gymnasium(Environment):
    '''
    A wrapper around gymanasium environments.

    Initialiization parameters and their descriptions mirror that of `gymnasium.make`.

    *Details can be found at the [Gymnasium docs [API > Make and register]](https://gymnasium.farama.org/api/registry/).*
    
    **Forward I/O**: ["obs", "act", "episode", "step"] -> ["rew", "next_obs", "done", "episode", "step"]
    '''

    def __init__(
        self, 
        id: str | EnvSpec,
        max_episode_steps: int | None = None,
        disable_env_checker: bool | None = None,
        **kwargs
    ):
        '''
        Initialize the Gymnasium environment.

        Parameters
        ----------
        id : str | EnvSpec
            The environment ID, or an `EnvSpec` object. Optionally, if using a string, a module 
            to import can be included, e.g. `module:Env-v0`. This is equivalent to importing the 
            module first to register the environment, followed by making the environment.
        max_episode_steps : int | None, optional
            Maximum length of an episode. Can override the registered `max_episode_steps` value
            of `EnvSpec` with the value being passed to `gymnasium.wrappers.TimeLimit`. Using 
            `max_episode_steps=-1` will not apply the wrapper to the environment.
        disable_env_checker : bool | None, optional
            If set, disables `gymnasium.wrappers.PassiveEnvChecker`, which wraps 
            `step`, `reset` and `render` functions to ensure compatibility with the 
            Gymnasium API. Defaults to the `disable_env_checker` value of `EnvSpec`.
        **kwargs : dict, optional
            Additional arguments to pass to the environment constructor.
        '''
        # Initialize the gymnasium environment
        self.env = gymnasium.make(
            id, 
            max_episode_steps=max_episode_steps, 
            disable_env_checker=disable_env_checker, 
            **kwargs
        )

        # Parse the observation space
        # self.obs_shape, _, _, self.obs_dtype = self._parse_space(self.env.observation_space)
        # if len(self.obs_shape) > 1 and self.obs_dtype == torch.float64:
        #     obs_shape = torch.Size((self.obs_shape.numel()/self.obs_shape[-1], self.obs_shape[-1]))
        # else:
        #     obs_shape = torch.Size((self.obs_shape.numel(),))

        # Parse the action space
        # self.act_shape, act_min, act_max, self.act_dtype = self._parse_space(self.env.action_space)
        # if len(self.act_shape) > 1 and self.act_dtype == torch.float64:
        #     act_shape = torch.Size((self.act_shape.numel()/self.act_shape[-1], self.act_shape[-1]))
        # else:
        #     act_shape = torch.Size((self.act_shape.numel(),))

        obs_space_flat = flatten_space(self.env.observation_space)
        assert isinstance(obs_space_flat, spaces.Box), \
            "Gymnasium wrapper currently only supports observation spaces that can be flattened to a Box space."

        act_space_flat = flatten_space(self.env.action_space)
        assert isinstance(act_space_flat, spaces.Box), \
            "Gymnasium wrapper currently only supports action spaces that can be flattened to a Box space."

        act_low = torch.tensor(act_space_flat.low, dtype=torch.float32)
        act_high = torch.tensor(act_space_flat.high, dtype=torch.float32)
        assert all([act_low[0] == m for m in act_low]) and all([act_high[0] == m for m in act_high]), \
            "Heterogeneous bounds for Box action spaces are not currently supported."
        
        self.obs_shape = torch.Size(obs_space_flat.shape)
        self.act_shape = torch.Size(act_space_flat.shape)
        
        # Initialize the base Environment class with a prototype experience
        super().__init__(
            TensorDict({
                "obs": torch.zeros(self.obs_shape, dtype=torch.float32),
                "act": torch.zeros(self.act_shape, dtype=torch.float32),
                "rew": torch.zeros((1,), dtype=torch.float32),
                "next_obs": torch.zeros(self.obs_shape, dtype=torch.float32),
                "done": torch.zeros((1,), dtype=torch.float32),
                "act_min": act_low,
                "act_max": act_high,
            })
        )

        self.act_out = self.act_out = TensorDictModule(lambda x: x, in_keys=["act"], out_keys=["act_out"])
        if self.env.action_space.dtype == np.int64:
            self.act_out = TensorDictModule(
                lambda act: F.one_hot(torch.argmax(act, dim=-1), num_classes=self.act_shape[-1]),
                in_keys=["act"], out_keys=["act_out"]
            )

        # Define an attribute to store complementary info returned by environments
        self.info = None

    def reset(self, experience: TensorDict) -> TensorDict:
        '''
        Reset the environment and return the initial observation.

        Parameters
        ----------
        experience : TensorDict
            The experience to be updated with the initial observation.

        Returns
        -------
        experience : TensorDict
            With new or modified keys:
            - "obs": The initial observation from the environment.
            - "episode": Incremented episode count.
            - "step": Step count reset to 0.
        '''
        obs, info = self.env.reset()
        obs = flatten(self.env.observation_space, obs)

        experience.set("next_obs", torch.tensor(obs, dtype=torch.float32), inplace=True)
        self.info = info

        experience.set("done", torch.zeros((1,), dtype=torch.float32), inplace=True)
        experience.set("step", torch.zeros((1,), dtype=torch.int64), inplace=True)
        experience["episode"] += 1

        return experience
    
    def step(self, experience: TensorDict) -> TensorDict:
        '''
        Take a step in the environment using the current action.

        Parameters
        ----------
        experience : TensorDict
            With existing keys:
            - "act": The action to be taken in the environment.
        
        Returns
        -------
        experience : TensorDict
            With new or modified keys:
            - "next_obs": The observation after taking the action.
            - "rew": The reward received after taking the action.
            - "done": A boolean indicating if the episode has ended (True if terminated or truncated).
            - "step": Incremented step count.
        '''
        action = self.act_out(experience)["act_out"].cpu().numpy()
        action = unflatten(self.env.action_space, action)

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs = flatten(self.env.observation_space, next_obs)

        experience.set("next_obs", torch.tensor(next_obs, dtype=torch.float32), inplace=True)
        experience.set("rew", torch.tensor(reward, dtype=torch.float32).unsqueeze(0), inplace=True)
        experience.set("done", torch.tensor(terminated or truncated, dtype=torch.float32).unsqueeze(0), inplace=True)
        self.info = info

        experience["step"] += 1

        return experience.exclude("act_out", inplace=True)

    def render(self) -> None:
        '''
        Render the current state of the environment.

        Parameters
        ----------
        mode : str, optional
            The mode in which to render the environment. Default is "human".
        '''
        self.env.render()

    def close(self) -> None:
        '''
        Close the environment and release any resources.
        '''
        self.env.close()
