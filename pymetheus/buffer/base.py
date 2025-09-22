
from abc import ABC, abstractmethod
import torch
from tensordict import TensorDict   # type: ignore


class Buffer(ABC):
    '''
    Abstract Base Class for experience replay/rollout buffers.

    Stores experiences to be used for training. In the on-policy setting,
    rollout buffers store experiences from the current policy for immediate
    training and are cleared after each training step. In the off-policy
    setting, replay buffers store experiences from past policies and are used
    for training according to some sampling strategy; when the buffer reaches
    capacity, experiences are evicted according to some eviction policy.

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

    Abstract Attributes
    -------------------
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
    **sample**(k[int | None] -> tuple[list[int], list[float], TensorDict])
        Sample a batch of experiences from the buffer.
    **update**(idcs[list[int], priorities[list[float]])
        Update the priorities of experiences at the specified indices.

    Abstract Methods
    ----------------
    **_clear**()
        Reset the buffer to an empty state.
    **_add**(float, TensorDict)
        Push a new experience with the given priority into the storage.
    **_update**(list[int], list[float])
        Replace the items at the specified indices with new priorities.
    **_sample**() -> TensorDict
        Sample a batch of experiences from the buffer.

    Static Methods
    --------------
    **calc_capacity**(TensorDict, int -> int)
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
        device: str = "cpu"
    ):
        '''
        Initialize the experience replay buffer.

        Parameters
        ----------
        storage : Storage
            The storage backend for the buffer, which manages the
            experiences and their priorities.
        batch_size : int
            The number of experiences to sample from the buffer at once.
        priority_key : str, optional
            The key used for priority sampling. If not specified, the priority
            defaults to the product of `episode` and `step` values, resulting
            in a FIFO replacement policy.
        '''
        self._proto_experience = proto_experience
        self._capacity = capacity
        self._batch_size = batch_size
        self._priority_key = priority_key
        self._device = device
        self._storage = TensorDict(
            proto_experience.clone().to(device).expand(capacity),
            batch_size=[capacity],
            device=device
        )

    @property
    def proto_experience(self) -> TensorDict:
        ''' Prototype experience used to initialize the storage. '''
        return self._proto_experience

    @property
    def capacity(self) -> int:
        ''' Maximum number of experiences the storage can hold. '''
        return self._capacity

    @property
    def batch_size(self) -> int:
        ''' The batch size used for sampling experiences from the buffer. '''
        return self._batch_size

    @property
    def priority_key(self) -> str | None:
        ''' The key used for priority sampling. '''
        return self._priority_key

    @property
    def device(self) -> str:
        ''' Device where the data is stored and processed. '''
        return self._device

    @property
    def storage(self) -> TensorDict:
        ''' The storage TensorDict containing the raw experience data. '''
        return self._storage

    @property
    @abstractmethod
    def total(self) -> float:
        ''' Total sum of priorities in the storage. '''
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        ''' Maximum priority value in the storage. '''
        pass

    @abstractmethod
    def _clear(self):
        ''' Reset the buffer to an empty state. '''
        pass

    def clear(self):
        ''' Clear the buffer and reset its state. '''
        proto_experience = self._proto_experience.clone().to(self._device)
        self._storage = TensorDict(
            proto_experience.expand(self._capacity),
            batch_size=[self._capacity],
            device=self._device
        )
        self._clear()

    @abstractmethod
    def _add(self, priority: float, experience: TensorDict):
        '''
        Push a single experience with the given priority into the storage.

        Parameters
        ----------
        priority : float
            The priority of the experience.
        experience : TensorDict
            The experience to be stored.
        '''
        pass

    def add(self, experience: TensorDict):
        '''
        Add an experience to the buffer.

        Parameters
        ----------
        experience : TensorDict
            The experience to be added to the buffer.
        '''
        # Put experience on the correct device
        experience = experience.detach().to(self._device)

        if not self._priority_key:
            # Set FIFO priority
            priority = experience["episode"] * experience["step"]
        elif self._priority_key not in experience:
            # Set default priority
            priority = torch.full_like(experience["step"], self.max)
        else:
            # Get priority from the specified key
            priority = experience[self._priority_key]
        priority = torch.abs(priority)

        # Ensure experience contains only required keys
        experience.select(*self._proto_experience.keys(), inplace=True)

        if len(experience["step"].shape) == 1:
            if experience["step"] > 0:
                # Add experience to the storage
                self._add(priority.item(), experience)
        elif experience["step"].shape[0] == self._batch_size:
            # Set batch dimension
            experience.batch_size = [self._batch_size]
            # Add experiences to the storage
            for p, e in zip(priority, experience):
                self._add(p.item(), e)

    @abstractmethod
    def _sample(self, k: int) -> tuple[TensorDict,
                                       list[int], list[float]]:
        '''
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        k : int
            The number of experiences to sample.

        Returns
        -------
        samples : TensorDict
            The sampled experiences.
        indices : list[int]
            The indices of the sampled experiences in the buffer.
        priorities : list[float]
            The priorities of the sampled experiences.
        '''
        pass

    def sample(self, k: int | None = None) -> tuple[TensorDict,
                                                    list[int], list[float]]:
        '''
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        k : int, optional
            The number of experiences to sample. If not specified,
            defaults to the buffer's batch size.

        Returns
        -------
        samples : TensorDict
            The sampled experiences.
        indices : list[int]
            The indices of the sampled experiences in the buffer.
        priorities : list[float]
            The priorities of the sampled experiences.
        '''
        return self._sample(k or self._batch_size)

    @abstractmethod
    def _update(self, idcs: list[int], priorities: list[float]):
        '''
        Replace the items at the specified indices with new priorities.

        Parameters
        ----------
        idcs : list[int]
            Indices of the items to replace.
        priorities : list[float]
            New priority values for the items.
        '''
        pass

    def update(self, idcs: list[int], priorities: list[float] | TensorDict):
        '''
        Update the priorities of experiences at the specified indices.

        Parameters
        ----------
        idcs : list[int]
            Indices of the items to replace.
        priorities : list[float] | TensorDict
            New priority values for the items. If a TensorDict is provided,
            it should contain the priority key.
        '''
        if isinstance(priorities, TensorDict):
            priorities = priorities[self._priority_key]
            priorities = priorities.to(self._device).flatten().tolist()
        # Get unique indices and their maximum priorities
        unique: dict[int, float] = {}
        for idx, priority in zip(idcs, priorities):
            priority = abs(priority)
            if idx not in unique or priority > unique[idx]:
                unique[idx] = priority
        idcs, priorities = list(unique.keys()), list(unique.values())
        self._update(idcs, priorities)

    @staticmethod
    def calc_capacity(proto_experience: TensorDict, megabytes: int) -> int:
        '''
        Calculate the number of experiences that can be stored
        in the buffer based on the available physical storage.

        Parameters
        ----------
        proto_experience : TensorDict
            A zero-TensorDict defining the keys and shapes of the experience.
        megabytes : int
            The total available memory allocation for the buffer in megabytes.

        Returns
        -------
        capacity : int
            The maximum number of experiences that can be stored in the buffer.
        '''
        return (megabytes * 1024 * 1024) // proto_experience.bytes()
