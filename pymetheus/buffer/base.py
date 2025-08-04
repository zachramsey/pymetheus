
import torch
from tensordict import TensorDict
from ..utils import TensorMinMaxHeap

class Buffer:
    '''
    Abstract Base Class for experience replay/rollout buffers.  
    Stores experiences to be used for training. In the on-policy setting, rollout 
    buffers store experiences from the current policy for immediate training and 
    are cleared after each training step. In the off-policy setting, replay buffers 
    store experiences from past policies and are used for training according to some 
    sampling strategy; when the buffer reaches capacity, experiences are evicted 
    according to some eviction policy (e.g., FIFO).

    Attributes
    ----------
    batch_size : int
        The batch size used for sampling experiences from the buffer.
    capacity : int
        The maximum capacity of the replay buffer.
    count : int
        The current number of experiences in the buffer.
    storage : TensorDict
        The storage of the replay buffer, containing the experiences.
    
    Methods
    -------
    **add**(experience[TensorDict])
        Add a single experience or a batch of experiences to the buffer.
    **sample**() -> batch[TensorDict]
        Sample a batch of experiences from the buffer. This method may be
        overridden by subclasses to implement different sampling strategies.
    **clear**()
        Clear the buffer, resetting it to an empty state.

    Static Methods
    --------------
    **calc_capacity**(TensorDict, int -> int)
        Calculate the number of experiences that can be stored in the buffer based on 
        the available physical storage. This is useful for determining the buffer's 
        capacity based on the size of the experiences and the available memory.
    '''

    def __init__(
        self, 
        proto_experience: TensorDict, 
        capacity: int, 
        batch_size: int, 
        priority_key: str = None
    ):
        '''
        Initialize the experience replay buffer.

        Parameters
        ----------
        proto_experience : TensorDict
            A TensorDict of zeroes that defines the keys and Tensor shapes of the experience.
        capacity : int
            The maximum number of experiences the buffer can store.
        batch_size : int
            The number of experiences to sample from the buffer at once.
        priority_key : str, optional
            The key used for priority sampling. If not specified, the priority defaults to the 
            product of `episode` and `step` values, resulting in a FIFO replacement policy.
        '''
        self._proto_experience = proto_experience
        self._batch_size = batch_size
        self._priority_key = priority_key
        self._storage = TensorMinMaxHeap(proto_experience, capacity, priority_key)

    def __len__(self) -> int:
        ''' Current number of experiences in the buffer. '''
        return len(self._storage)

    def add(self, experience: TensorDict):
        '''
        Add an experience to the buffer.

        Parameters
        ----------
        experience : TensorDict
            The experience to be added to the buffer.  
            *Must match the structure defined by `proto_experience`.*
        '''
        # Validate the incoming experience structure against proto_experience
        assert isinstance(experience, TensorDict), \
            f"Experience must be a TensorDict, got {type(experience)}."
        
        # Check if the experience has the same keys as proto_experience
        assert experience.keys() == self._proto_experience.keys(), \
            "Experience keys do not match proto_experience keys."

        # Check if the shapes of the experience's tensors are compatible with proto_experience
        for key in self._proto_experience.keys():
            assert self._proto_experience[key].shape == experience[key].shape, \
                f"Shape mismatch for key '{key}': expected {self._proto_experience[key].shape}, got {experience[key].shape}."

        # Discard the initial environment state
        if experience["step"] > 0:
            # Set default priority if not specified
            if not self._priority_key:
                experience["priority"] = experience["episode"] * experience["step"]

            # Push the experience into the buffer
            self._storage.push(experience)

    def sample(self) -> TensorDict:
        '''
        Sample a batch of experiences from the buffer.
        
        *By default, if `priority_key` is not specified, experiences are sampled uniformly at random;
        otherwise, experiences are sampled in priority order wrt. their value at `priority_key`.*

        Returns
        -------
        batch : TensorDict
            A batch of experiences sampled from the buffer.
        
        Notes
        -----
        This method can be overridden by subclasses to implement different sampling strategies.
        '''
        batch = TensorDict(self._proto_experience.clone().expand(self._batch_size))
        if self._priority_key:
            for i in range(self._batch_size):
                batch[i] = self._storage.pop_max()
        else:
            rand = torch.randperm(self._batch_size)
            for i in range(self._batch_size):
                batch[i] = self._storage[rand[i]]
        return batch
    
    def clear(self):
        '''
        Clear the buffer, resetting it to an empty state.
        '''
        self._storage.clear()

    @staticmethod
    def calc_capacity(proto_experience: TensorDict, megabytes: int) -> int:
        '''
        Calculate the number of experiences that can be stored 
        in the buffer based on the available physical storage.

        Parameters
        ----------
        proto_experience : TensorDict
            A TensorDict of zeroes that defines the keys and Tensor shapes of the experience.
        megabytes : int
            The total available memory allocation for the buffer in megabytes.

        Returns
        -------
        capacity : int
            The maximum number of experiences that can be stored in the buffer.
        '''
        return (megabytes * 1024 * 1024) // proto_experience.bytes()

