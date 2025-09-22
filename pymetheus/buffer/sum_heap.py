
from . import Buffer
import random
from tensordict import TensorDict   # type: ignore


class SumHeap(Buffer):
    '''
    Hybrid data structure combining a MinHeap and SumTree, enabling
    both efficient replacement of the minimum priority item when the
    storage is full and efficient priority-based sampling and updates.

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
        self._tree = [0.0] * (capacity-1)   # Track total priorities
        self._heap = [(0.0, -1)] * capacity  # Minimum priority heap
        self._size = 0                      # Num experiences in the heap
        self._max = 1.0                     # Maximum priority in the heap
        self._updates = 0                   # Num updates since last heapify

    def __len__(self):
        return self._size

    @property
    def total(self):
        return self._tree[0]

    @property
    def max(self):
        return self._max

    def __set(self, idx: int, item: tuple[float, int]):
        ''' Set the `item` at `idx` in the min-heap and update the SumTree. '''
        # Get the change in priority
        diff = item[0] - self._heap[idx][0]
        # Set the item in the heap
        self._heap[idx] = item
        # Propagate change up, stop when root reached
        idx += self._capacity - 1
        while idx > 0:
            idx = (idx - 1) >> 1
            self._tree[idx] += diff

    def __push_up(self, idx: int, start: int = 0):
        '''
        Pushes the item at `idx` up to its correct position in the heap.\n
        *Adapted from Python's heapq module.*
        '''
        item = self._heap[idx]
        while idx > start:
            par_idx = (idx - 1) >> 1
            par_item = self._heap[par_idx]
            if item < par_item:
                self.__set(idx, par_item)
                idx = par_idx
                continue
            break
        self.__set(idx, item)

    def __push_down(self, idx: int):
        '''
        Pushes the item at `idx` down to its correct position in the heap.\n
        *Adapted from Python's heapq module.*
        '''
        start = idx
        item = self._heap[idx]
        # Bubble up the smaller child until hitting a leaf.
        child_idx = 2 * idx + 1
        while child_idx < self._size:
            # Select the smaller child
            right_idx = child_idx + 1
            if (right_idx < self._size):
                if not (self._heap[child_idx] < self._heap[right_idx]):
                    child_idx = right_idx
            # Move the smaller child up.
            self.__set(idx, self._heap[child_idx])
            idx = child_idx
            child_idx = 2 * idx + 1
        # The leaf at idx is empty now.  Put item there, and bubble it up
        # to its final resting place (by pushing its parents up).
        self.__set(idx, item)
        self.__push_up(idx, start)

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
            - The index of the item in the heap.
            - The priority of the item.
            - The data associated with the item.
        '''
        idx = 0
        while True:
            idx = 2 * idx + 1                       # Move to left child
            if idx < self._capacity-1:
                # Child is an internal node
                val = self._tree[idx]               # Left child's sum
                if threshold > val:
                    # Left priority insufficient
                    idx += 1                        # Move to right child
                    threshold -= val                # Subtract left sum
            else:
                # Child is a leaf node
                idx -= self._capacity-1             # Convert to min-heap index
                val = self._heap[idx][0]            # Left child's priority
                if threshold > val:
                    # Left priority insufficient
                    idx += 1                        # Move to right child
                priority, data_idx = self._heap[idx]
                return idx, priority, self._storage[data_idx]

    def _clear(self):
        self._tree = [0.0] * (self._capacity - 1)
        self._heap = [(0.0, -1)] * self._capacity
        self._size = 0
        self._max = 1.0
        self._updates = 0

    def _add(self, priority: float, experience: TensorDict):
        # Update the maximum priority
        self._max = max(self._max, priority)
        # Add the new experience to the heap
        if self._size < self._capacity:
            # Heap is not full -> Append new experience
            self._storage[self._size] = experience.clone()  # Store experience
            self.__set(self._size, (priority, self._size))  # Add item to heap
            self.__push_up(self._size)                      # Push up new item
            self._size += 1                                 # Increase the size
        else:
            # Heap is full -> Replace the root
            min_priority, idx = self._heap[0]       # Get min priority
            if priority <= min_priority:
                return                              # New priority too small
            self._storage[idx] = experience.clone()  # Store the new experience
            self.__set(0, (priority, idx))          # Replace root w/ new item
            self.__push_down(0)                     # Push down new root item

    def _sample(self, k: int) -> tuple[list[int], list[float], TensorDict]:
        indices = []
        priorities = []
        samples = TensorDict(
            self._proto_experience.clone().expand([k]),
            batch_size=[k],
            device=self._device
        )
        seg = self._tree[0] / k
        for i in range(k):
            threshold = random.uniform(i*seg, (i+1)*seg)
            idx, priority, data = self.__get(threshold)
            indices.append(idx)
            priorities.append(priority)
            samples[i] = data
        return indices, priorities, samples

    def _update(self, idcs: list[int], priorities: list[float]):
        # Update the maximum priority
        self._max = max(priorities + [self._max])
        # Update each priority in the batch
        do_heapify = False
        for idx, priority in zip(idcs, priorities):
            self.__set(idx, (priority, self._heap[idx][1]))
            # Periodically restore the heap property
            self._updates = (self._updates + 1) % self._heapify_period
            if self._updates == 0:
                do_heapify = True
        if do_heapify:
            for i in reversed(range(self._size//2)):
                self.__push_down(i)
