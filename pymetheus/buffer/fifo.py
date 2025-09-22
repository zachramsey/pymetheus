from . import Buffer
from random import randint
from tensordict import TensorDict   # type: ignore


class FIFO(Buffer):
    '''
    Experience buffer that evicts experiences in a first-in,
    first-out manner and samples experiences uniformly at random.

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
        self._next = 0

        # NOTE: maintained for compliance; not currently used
        self._priorities = [0.0] * capacity
        self._total = 0.0
        self._max = 1.0

    def __len__(self) -> int:
        return self._size

    @property
    def total(self) -> float:
        return self._total

    @property
    def max(self) -> float:
        return self._max

    def _clear(self):
        self._priorities = [0.0] * self._capacity
        self._size = 0
        self._next = 0
        self._total = 0.0
        self._max = 1.0

    def _add(self, priority: float, data: TensorDict):
        self._max = max(self._max, priority)
        self._total += priority - self._priorities[self._next]
        self._priorities[self._next] = priority
        self._storage[self._next] = data.clone()
        self._next = (self._next + 1) % self._capacity
        self._size += not self._size == self._capacity

    def _sample(self, k: int | None = None) -> tuple[list[int], list[float],
                                                     TensorDict]:
        k = k or self._batch_size
        indices = []
        priorities = []
        samples = TensorDict(
            self._proto_experience.clone().expand([k]),
            batch_size=[k],
            device=self._device
        )
        for i in range(k):
            idx = randint(0, self._size - 1)
            indices.append(idx)
            priorities.append(self._priorities[idx])
            samples[i] = self._storage[idx].clone()
        return indices, priorities, samples

    def _update(self, idcs: list[int], priorities: list[float]):
        self._max = max(priorities + [self._max])
        self._total += sum(priorities) - sum(self._priorities[idx]
                                             for idx in idcs)
        for idx, priority in zip(idcs, priorities):
            self._priorities[idx] = priority
