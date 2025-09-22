
from . import Buffer
from random import randint
from tensordict import TensorDict   # type: ignore


class Uniform(Buffer):
    '''
    Experience buffer that evicts experiences uniformly at random.

    Attributes
    ----------
    proto_experience : TensorDict
        A prototype experience used to initialize the storage.
    capacity : int
        The maximum number of experiences the storage can hold.
    storage : TensorDict
        The storage of the experiences.
    batch_size : int
        The batch size used for sampling experiences from the buffer.
    count : int
        The current number of experiences in the buffer.
    priority_key : str | None
        The key used for priority sampling.
    device : str
        The device on which the storage operates (e.g., "cpu", "cuda").
    total : float
        Total sum of priorities in the storage.
    max : float
        Maximum priority value in the storage.

    Methods
    -------
    **clear**()
        Clear the buffer, resetting it to an empty state.
    **add**(experience[TensorDict])
        Add a single experience or a batch of experiences to the buffer.
    **sample**(int | None) -> list[int], list[float], TensorDict
        Sample a batch of indices, priorities, & experiences from the buffer.\n
        *May be overridden in subclasses for different sampling strategies.*
    **update**(idcs[list[int], priorities[list[float]])
        Update the priorities of experiences at the specified indices.

    Static Methods
    --------------
    **calc_capacity**(TensorDict, int) -> int
        Calculate the number of experiences that can be stored in the
        buffer based on the available physical storage. This is useful
        for determining the buffer's capacity based on the size of the
        experiences and the available memory.
    '''

    def __init__(
        self,
        proto_experience: TensorDict,
        capacity: int,
        batch_size: int,
        priority_key: str | None = None,
        device: str = "cpu",
    ):
        '''
        Initializes the TensorHeap.

        Parameters
        ----------
        proto_experience : TensorDict
            A prototype experience used to initialize the heap's storage.
        capacity : int
            The maximum number of experiences the heap can hold.
        batch_size : int
            The batch size used for sampling experiences from the buffer.
        priority_key : str, optional
            The key used for priority sampling.
        device : str, optional
            The device on which the heap will operate.
        '''
        super().__init__(proto_experience,
                         capacity,
                         batch_size,
                         priority_key,
                         device)
        self._size = 0

        # NOTE: maintained for compliance; not currently used
        self._priorities = [0.0] * capacity
        self._total = 0.0
        self._max = 1.0

    def __len__(self):
        return self._size

    @property
    def total(self):
        return self._total

    @property
    def max(self):
        return self._max

    def _clear(self):
        self._priorities = [0.0] * self._capacity
        self._size = 0
        self._total = 0.0
        self._max = 1.0

    def _add(self, priority, experience):
        self._max = max(self._max, priority)
        full = self._size == self._capacity
        idx = randint(0, len(self._priorities) - 1) if full else self._size
        self._total += priority - self._priorities[idx]
        self._priorities[idx] = priority
        self._storage[idx] = experience.clone()
        self._size += (not full)

    def _sample(self, k):
        '''
        Sample a batch of experiences from the buffer using random sampling.

        Returns
        -------
        indices : list[int]
            The indices of the sampled experiences in the buffer.
        priorities : list[float]
            The priorities of the sampled experiences.
        samples : TensorDict
            The sampled experiences.
        '''
        indices = []
        priorities = []
        samples = TensorDict(
            self._proto_experience.clone().expand([k]),
            batch_size=[k],
            device=self._device
        )
        for i in range(k):
            idx = randint(0, len(self._storage) - 1)
            indices.append(idx)
            priority = self._priorities[idx]
            priorities.append(priority)
            samples[i] = self._storage[idx].clone()
        return indices, priorities, samples

    def _update(self, idcs, priorities):
        self._max = max(priorities + [self._max])
        self._total += sum(priorities) - sum(self._priorities[idx]
                                             for idx in idcs)
        for idx, priority in zip(idcs, priorities):
            self._priorities[idx] = priority
