
from ..reset import Reset
from ..transition import Transition
from ..reward import Reward
from ..terminate import Terminate

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
import torch

class Environment(TensorDictModuleBase):
    '''
    Abstract Base Class for reinforcement learning environments and environment models.

    This class provides a framework for defining (external) environments and (internal) 
    environment models for reinforcement learning tasks. A training loop may interact 
    with an environment to update experiences through familar calls to `reset` and `step` 
    methods. However, this class also provides a `forward` method that simplifies this 
    process and makes environments handle like PyTorch modules.

    Invoking the environment `forward` method in the training loop obviates the need for explicit 
    `reset` and `step` calls. This approach will step through the environment, updating the 
    experience as usual, except on the initial step and when the done flag is set; here, the 
    environment is automatically reset and the initial observation is returned in an otherwise 
    empty experience (as defined by the `proto_experience` TensorDict).

    An environment or environment model may be defined in two ways:
    - Subclassing `Environment` and directly overriding the `reset` and `step` methods.
    - Passing subclass instances of `Reset`, `Transition`, `Reward`, and `Terminate` to 
      the constructor of `Environment` or a subclass thereof.

    **Forward I/O**: ["obs", "action", "episode", "step"] -> ["next_obs", "reward", "done", "episode", "step"]

    Methods
    -------
    **reset**(proto_experience[TensorDict]) -> experience[TensorDict]
        Reset the environment and return the initial observation.
        Automatically sets the done flag to 0 (not done) after reset.
    **step**(experience[TensorDict]) -> experience[TensorDict]
        Take a step in the environment using the provided experience.
        Updates next_obs, reward, and done flag in the experience.
    **forward**(experience[TensorDict]) -> experience[TensorDict]
        Take an action in the environment and update the experience.
        Automatically resets the environment if the done flag is set.
        
    Abstract Methods
    ----------------
    **render**(mode[str])
        Render the environment.
    **close**()
        Close the environment, releasing any external resources.
    '''

    def __init__(
        self,
        proto_experience: TensorDict,
        reset: Reset = None,
        transition: Transition = None,
        reward: Reward = None,
        terminate: Terminate = None
    ):
        '''
        Initialize the environment.

        Parameters
        ----------
        proto_experience : TensorDict
            TensorDict of zeroes that defines the keys and Tensor shapes of the experience.  
            Expects at least the following keys:
            - **obs**: The current observation of the environment.
            - **act**: The action taken by the agent.
            - **next_obs**: The observation of the environment after taking the action.
            - **rew**: The reward received after taking the action.
            - **done**: A flag indicating if the episode has ended.
            - **act_bounds**: The minimum and maximum bounds for the action space.
        reset : Reset, optional
            A callable instance of the `Reset` class that defines how to reset the environment.
        transition : Transition, optional
            A callable instance of the `Transition` class that defines how to transition the environment
            from one state to another based on the action taken.
        reward : Reward, optional
            A callable instance of the `Reward` class that defines how to compute the reward based on
            the current observation and action.
        terminate : Terminate, optional
            A callable instance of the `Terminate` class that defines how to determine if the episode has
            ended based on the current observation and action.
        device : torch.device, optional
            The device on which the environment should operate,  
        '''
        super().__init__()
        self.proto_experience = proto_experience

        # Check that proto_experience is a TensorDict
        assert isinstance(self.proto_experience, TensorDict), \
            f"Expected proto_experience to be a TensorDict, got {type(self.proto_experience)}"
        
        # Check that proto_experience contains the required keys
        for key in ['obs', 'act', 'rew', 'next_obs', 'done', 'act_min', 'act_max']:
            assert key in self.proto_experience.keys(), \
                f"Expected proto_experience to contain key '{key}'"
            
        # Add 'episode' and 'step' keys to proto_experience
        self.proto_experience.set("episode", torch.zeros((1,), dtype=torch.int64) - 1, inplace=True)
        self.proto_experience.set("step", torch.zeros((1,), dtype=torch.int64), inplace=True)

        # Set the done flag to invoke initial reset
        self.proto_experience.set("done", torch.ones((1,), dtype=torch.float32), inplace=True)
            
        # Set the input keys to match the prototype experience
        self.in_keys = list(self.proto_experience.keys())
        self.out_keys = list(self.proto_experience.keys())

        # Set the reset function
        self._reset = reset

        # Set the transition, reward, and terminate functions
        self._transition = transition
        self._reward = reward
        self._terminate = terminate

        # Define the step function as a TensorDictSequential
        self._step = None
        if (isinstance(self._transition, Transition) and
            isinstance(self._reward, Reward) and
            isinstance(self._terminate, Terminate)):
            self._step = TensorDictSequential(transition, reward, terminate)

    # HACK: Ensure proto_experience is moved to device with the environment
    # TODO: Replace this workaround with a native solution
    # NOTE: Possible fix by wrapping proto_experience in TensorDictParams
    def to(self, *args, **kwargs):
        self.proto_experience = self.proto_experience.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def reset(self, proto_experience: TensorDict) -> TensorDict:
        '''
        Reset the environment and return the initial observation.

        *This method may be defined either by directly overriding in an Environment 
        subclass or by passing a `Reset` subclass instance to the constructor.*

        Parameters
        ----------
        proto_experience : TensorDict
            Prototype experience TensorDict, which defines the keys and shapes of the experience.

        Returns
        -------
        experience : TensorDict
            With new or modified items:
            - "obs": Initial observation of the environment.
            - "done": Termination flag set to 0 (not done).
        '''
        # Reset the environment
        try:
            experience = self._reset(proto_experience)
        except TypeError as e:
            if self._reset is None or not callable(self._reset):
                # If reset is not defined, print an explicit error message
                print("Error: `reset` is not defined! Callable must be \
                       provided in the environment constructor, OR the \
                       `reset` method must be overriden by a subclass.")
            else:
                # Otherwise, raise the original error
                raise e

        # Ensure the termination flag is set to 0 (not done) after reset
        experience.set("done", torch.zeros((1,), dtype=torch.float32), inplace=True)

        # Reset the step count
        experience.set("step", torch.zeros((1,), dtype=torch.int64), inplace=True)

        # Increment the episode count
        experience["episode"] += 1

        return experience
    
    def step(self, experience: TensorDict) -> TensorDict:
        '''
        Take a step in the environment using the provided experience.

        *This method may be defined either by directly overriding it in an Environment subclass or 
        by passing `Transition`, `Reward`, and `Terminate` subclass instances to the constructor.*

        Parameters
        ----------
        experience : TensorDict
            With existing items:
            - "obs": Current observation of the environment.
            - "action": Action taken by the agent.

        Returns
        -------
        experience : TensorDict
            With new or modified items:
            - "next_obs": Observation of the environment after taking the action.
            - "reward": Reward received after taking the action.
            - "done": Flag indicating if the episode has ended (1 if done, 0 otherwise).
        '''
        try:
            experience = self._step(experience)
        except TypeError as e:
            if self._step is None:
                # If step is not defined, print an explicit error message
                none_func = ""
                if self._transition is None or not callable(self._transition):
                    none_func += " `transition`"
                if self._reward is None or not callable(self._reward):
                    none_func += " `reward`"
                if self._terminate is None or not callable(self._terminate):
                    none_func += " `terminate`"
                print(f"Error:{none_func} not defined! Callable(s) \
                        must be provided in the environment constructor, OR \
                        the `step` method must be overriden by a subclass.")
            else:
                # Otherwise, raise the original error
                raise e
            
        # Increment the step count
        experience['step'] += 1

        return experience

    def forward(self, experience: TensorDict) -> TensorDict:
        '''
        Take an action in the environment and update the experience.

        *This method handles both stepping through the environment
        and automatically resetting it when the done flag is set.*

        Parameters
        ----------
        experience : TensorDict
            With existing items:
            - "obs": Current observation of the environment.
            - "action": Action taken by the agent.

        Returns
        -------
        experience : TensorDict
            With new or modified items:
            - "next_obs": Observation of the environment after taking the action.
            - "reward": Reward received after taking the action.
            - "done": Flag indicating if the episode has ended (1 if done, 0 otherwise).
        '''
        if experience['done']>=0.5:                     # If the done flag is set,
            return self.reset(self.proto_experience)    # Reset the environment;
        else:                                           # Otherwise,
            return self.step(experience)                # Perform a step in the environment.

    def render(self, mode: str = 'human'):
        """
        Renders the environment.

        *This method is expected to be implemented in specialized `Environment` subclasses.*

        Parameters
        ----------
        mode : str, optional
            The mode in which to render the environment. Default is 'human'.
        """
        raise NotImplementedError("Render method must be implemented in a subclass.")

    def close(self):
        """
        Closes the environment, releasing any resources.

        *This method is expected to be implemented in specialized `Environment` subclasses.*
        """
        raise NotImplementedError("Close method must be implemented in a subclass.")
    