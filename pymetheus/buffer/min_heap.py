
from . import Buffer
import random
from tensordict import TensorDict   # type: ignore


class MinHeap(Buffer):
    '''
    A min-heap data structure for storing prioritized experiences.
    Provides efficient replacement of the minimum-priority experience and
    supports batch updates of priorities with periodic heapify.

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
        heapify_period: int | None = None
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
        heapify_period : int, optional
            How often to heapify the min-heap when batch-updating priorities.\n
            *Defaults to `capacity` steps.*
        '''
        super().__init__(proto_experience,
                         capacity,
                         batch_size,
                         priority_key,
                         device)
        self._heapify_period = heapify_period or capacity
        self._heap: list[tuple[float, int]] = []    # Min priority heap
        self._total = 0.0                           # Sum total of priorities
        self._max = 1.0                             # Maximum priority
        self._updates = 0                           # Updates from last heapify

    def __len__(self):
        return len(self._heap)

    @property
    def total(self):
        return self._total

    @property
    def max(self):
        return self._max

    def __push_up(self, idx, start=0):
        '''
        Pushes the item at `idx` up to its correct position in the heap.\n
        *Adapted from Python's heapq module.*
        '''
        item = self._heap[idx]
        while idx > start:
            par_idx = (idx - 1) >> 1
            par_item = self._heap[par_idx]
            if item < par_item:
                self._heap[idx] = par_item
                idx = par_idx
                continue
            break
        self._heap[idx] = item

    def __push_down(self, idx):
        '''
        Pushes the item at `idx` down to its correct position in the heap.\n
        *Adapted from Python's heapq module.*
        '''
        start = idx
        item = self._heap[idx]
        # Bubble up the smaller child until hitting a leaf.
        child_idx = 2 * idx + 1
        while child_idx < len(self._heap):
            # Select the smaller child
            right_idx = child_idx + 1
            if (right_idx < len(self._heap)):
                if not (self._heap[child_idx] < self._heap[right_idx]):
                    child_idx = right_idx
            # Move the smaller child up.
            self._heap[idx] = self._heap[child_idx]
            idx = child_idx
            child_idx = 2 * idx + 1
        # The leaf at idx is empty now. Put item there, bubble it up
        # to its final resting place (by pushing its parents up).
        self._heap[idx] = item
        self.__push_up(idx, start)

    def _clear(self):
        self._heap = []
        self._total = 0.0
        self._max = 1.0
        self._updates = 0

    def _add(self, priority: float, experience: TensorDict):
        self._max = max(self._max, priority)            # Update max priority
        self._total += priority                         # Update total priority
        if len(self._heap) < self._capacity:
            # Heap is not full -> Add new experience
            self._storage[len(self._heap)] = experience.clone()
            self._heap.append((priority, len(self._heap)))  # Add item to heap
            self.__push_up(len(self._heap)-1)               # Push up new item
        else:
            # Heap is full -> Replace the root
            min_priority, idx = self._heap[0]
            if priority <= min_priority:
                return                              # New priority too small
            self._total -= min_priority
            self._storage[idx] = experience.clone()  # Update stored experience
            self._heap[0] = (priority, idx)         # Replace root w/ new item
            self.__push_down(0)                     # Push down new item

    def _sample(self, k: int) -> tuple[list[int], list[float], TensorDict]:
        priorities = []
        samples = TensorDict(
            self._proto_experience.clone().expand([k]),
            batch_size=[k],
            device=self._device
        )
        # Rank-based sampling
        population = list(range(len(self._heap)))
        idcs = random.choices(population, weights=population, k=k)
        # Get the priorities and experiences for the sampled indices
        for i, idx in enumerate(idcs):
            priority, data_idx = self._heap[idx]
            priorities.append(priority)
            samples[i] = self._storage[data_idx]
        return idcs, priorities, samples

    def _update(self, idcs: list[int], priorities: list[float]):
        # Update the maximum priority and total sum of priorities
        self._max = max(priorities + [self._max])
        self._total += sum(priorities)-sum(self._heap[idx][0] for idx in idcs)
        # Update each priority in the batch
        do_heapify = False
        for idx, priority in zip(idcs, priorities):
            self._heap[idx] = (priority, self._heap[idx][1])
            # Periodically restore the min-heap property
            self._updates = (self._updates + 1) % self._heapify_period
            if self._updates == 0:
                do_heapify = True
        if do_heapify:
            for i in reversed(range(len(self._heap)//2)):
                self.__push_down(i)
