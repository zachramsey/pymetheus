
from . import Buffer
import random
from tensordict import TensorDict   # type: ignore
import torch


class SumTree(Buffer):
    '''
    A SumTree is a binary tree data structure that allows for efficient
    storage, updates, and retrieval of items based on their priorities.

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
        Sample a batch of indices, priorities, and experiences from the buffer.
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
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
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
        alpha : float, optional
            The exponent used to scale priorities.
        beta : float, optional
            The exponent used to scale importance-sampling weights.
        eps : float, optional
            A small constant added to priorities to ensure they are non-zero.
        '''
        super().__init__(proto_experience,
                         capacity,
                         batch_size,
                         priority_key,
                         device)
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self._tree = [0.0] * (2*capacity-1)     # Tracks total priorities
        self._size = 0                          # Num items in the SumTree
        self._next = 0                          # Next index to add a new item
        self._max = 1.0                         # Maximum priority in the tree

    def __len__(self) -> int:
        return self._size

    @property
    def total(self) -> float:
        return self._tree[0]

    @property
    def max(self) -> float:
        return self._max

    def __set(self, idx: int, priority: float):
        ''' Set the priority of the item at index `idx` to `priority`
            and update the tree accordingly. '''
        # Get index of leaf node
        idx += self._capacity - 1
        # Calc change in priority
        diff = priority - self._tree[idx]
        # Update the priority value
        self._tree[idx] = priority
        # Update the tree sums
        while idx > 0:
            idx = (idx - 1) >> 1
            self._tree[idx] += diff

    def __get(self, threshold: float) -> tuple[int, float, TensorDict]:
        '''
        Get the index, priority, and data of the item for which the sum of
        priorities in [0, index] is greater than or equal to `threshold`.

        Parameters
        ----------
        threshold : float
            The cumulative sum threshold to search for.

        Returns
        -------
        tuple[int, float, int]
            A tuple containing:
            - The index of the item in the SumTree.
            - The priority of the item.
            - The data associated with the item.
        '''
        idx = 0
        while idx < self._capacity - 1:
            idx = 2 * idx + 1                   # Move to left child
            if threshold > self._tree[idx]:
                # Threshold not satisfied by left child
                threshold -= self._tree[idx]
                idx += 1                        # Move to right child
        data_idx = idx - (self._capacity - 1)
        return data_idx, self._tree[idx], self._storage[data_idx]

    def _clear(self):
        self._tree = [0.0] * (2*self._capacity-1)
        self._size = 0
        self._next = 0
        self._max = 1.0

    def _add(self, priority: float, data: TensorDict):
        priority = (priority + self._eps) ** self._alpha
        # Update the maximum priority
        self._max = max(self._max, priority)
        # Store the experience data
        self._storage[self._next] = data.clone().to(self._device)
        # Update the priority at the next index
        self.__set(self._next, priority)
        # Update the count and next index
        self._size = min(self._size + 1, self._capacity)
        self._next = (self._next + 1) % self._capacity

    def _sample(self, k: int) -> tuple[TensorDict, list[int], list[float]]:
        indices = []
        priorities = []
        samples = TensorDict(
            self._proto_experience.clone().expand([k]),
            device=self._device
        )
        seg = self._tree[0] / k
        for i in range(k):
            threshold = random.uniform(seg*i, seg*(i+1))
            idx, priority, data = self.__get(threshold)
            indices.append(idx)
            priorities.append(priority)
            samples[i] = data
        weights = torch.tensor(priorities, device=self._device) / self.total
        weights = torch.pow(self._size * weights, -self._beta)
        weights /= weights.max()
        samples.set("weight", weights, inplace=True)
        return samples, indices, priorities

    def _update(self, idcs: list[int], priorities: list[float]):
        # Update the maximum priority
        self._max = max(priorities + [self._max])
        # Update each priority in the batch
        for idx, priority in zip(idcs, priorities):
            priority = (priority + self._eps) ** self._alpha
            self.__set(idx, priority)
