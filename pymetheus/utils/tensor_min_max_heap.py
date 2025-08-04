
from math import floor, log2
from tensordict import TensorDict

class TensorMinMaxHeap:
    '''
    Priority queue implemented as a min-max heap for storing, retrieving, and managing TensorDicts.

    Methods
    -------
    **clear**()
        Clears the min-max heap, resetting its storage.
    **push**(TensorDict)
        Push a new value into the min-max heap. If the heap is full, it replaces the minimum value.
    **pop**(int) -> TensorDict
        Pops the value at index `i` from the min-max heap.
    **pop_min**(int = 1) -> TensorDict
        Pops the minimum n values from the min-max heap.
    **pop_max**(int = 1) -> TensorDict
        Pops the maximum n values from the min-max heap.
    **peek_min**() -> TensorDict
        Peeks at the minimum value in the min-max heap without removing it.
    **peek_max**() -> TensorDict
        Peeks at the maximum value in the min-max heap without removing it.
    '''

    def __init__(
        self, 
        proto_experience: TensorDict, 
        capacity: int, 
        priority_key: str
    ):
        '''
        Initializes the TensorHeap.

        Parameters
        ----------
        proto_experience : TensorDict
            A prototype experience used to initialize the heap's storage.
        capacity : int
            The maximum number of experiences the heap can hold.
        priority_key : str | None, optional
            The key used for priority sampling. If not specified, defaults to "priority".
        '''
        self._proto_experience = proto_experience
        self._capacity = capacity
        self._priority_key = "priority" if priority_key is None else priority_key
        self._count = 0
        self._storage = TensorDict(proto_experience.clone().expand(capacity), batch_size=[capacity])

    def __call__(self):
        ''' Returns the current min-max heap as a TensorDict. '''
        return self._storage[:self._count]

    def __len__(self):
        ''' Returns the current size of the min-max heap. '''
        return self._count
    
    def __getitem__(self, index: int) -> TensorDict:
        '''
        Returns the value at the specified index of the contiguous heap storage.
        *Note that indexes are not in priority order for direct access.*
        '''
        if index >= self._count:
            raise IndexError(f"Index {index} out of bounds for heap of size {self._count}.")
        return self._storage[index].clone()
    
    @property
    def capacity(self):
        ''' Maximum capacity of the min-max heap. '''
        return self._capacity
    
    def clear(self):
        ''' Clears the min-max heap, resetting its storage. '''
        self._count = 0
        self._storage = TensorDict(self._proto_experience.clone().expand(self._capacity), batch_size=[self._capacity])
        
    def _push_down(self, m: int):
        '''
        Pushes the value at index `i` down the heap to maintain the min-max heap property.
        Called when a value is removed or replaced in the heap.
        '''
        # Loop until m has no children
        while True:
            i = m
            # Get indices of all descendants
            idcs = [idx for idx in [2*i+1, 2*i+2, 4*i+3, 4*i+4, 4*i+5, 4*i+6] if idx < self._count]
            # Break if there are no descendants
            if not idcs: break
            # Get index and value of each descendant
            descendants = [(idx, self._storage[idx]) for idx in idcs]
            # Get current value
            h_i = self._storage[i]

            # Index is on a min level
            if floor(log2(i + 1)) % 2 == 0:
                # Find the minimum descendant
                m, h_m = min(descendants, key=lambda x: x[1][self._priority_key])
                # Minimum descendant value is less than current value
                if h_m[self._priority_key] < h_i[self._priority_key]:
                    # Swap minimum descendant value with current value
                    self._storage[i], self._storage[m] = h_m.clone(), h_i.clone()
                    # Minimum descendant is a grandchild
                    if m >= 4 * i + 3:
                        p = (m - 1) // 2          # Minimum descendant parent index
                        h_p = self._storage[p]    # Minimum descendant parent value
                        # Minimum descendant value is greater than its parent value
                        if h_m[self._priority_key] > h_p[self._priority_key]:
                            # Swap minimum descendant with its parent
                            self._storage[m], self._storage[p] = h_p.clone(), h_m.clone()
                    else: break
                else: break

            # Index is on a max level
            else:
                # Find the maximum descendant
                m, h_m = max(descendants, key=lambda x: x[1][self._priority_key])
                # Maximum descendant value is greater than current value
                if h_m[self._priority_key] > h_i[self._priority_key]:
                    # Swap maximum descendant value with current value
                    self._storage[i], self._storage[m] = h_m.clone(), h_i.clone()
                    # Maximum descendant is a grandchild
                    if m >= 4 * i + 3:
                        p = (m - 1) // 2          # Maximum descendant parent index
                        h_p = self._storage[p]    # Maximum descendant parent value
                        # Maximum descendant value is less than its parent value
                        if h_m[self._priority_key] < h_p[self._priority_key]:
                            # Swap maximum descendant with its parent
                            self._storage[m], self._storage[p] = h_p.clone(), h_m.clone()
                    else: break
                else: break

    def _push_up_min(self, i: int):
        '''
        Pushes the value at index `i` up the heap to maintain the min-heap property.
        Used when the current index is on a min level.
        '''
        while True:
            gp = (i - 3) // 4
            # Grandparent index is valid
            if gp >= 0:
                h_i = self._storage[i]      # Current value
                h_gp = self._storage[gp]    # Grandparent value
                # Grandparent value is greater than current value
                if h_i[self._priority_key] < h_gp[self._priority_key]:
                    # Swap current value with grandparent value
                    self._storage[i], self._storage[gp] = h_gp.clone(), h_i.clone()
                    i = gp
                else: break
            else: break

    def _push_up_max(self, i: int):
        '''
        Pushes the value at index `i` up the heap to maintain the max-heap property.
        Used when the current index is on a max level.
        '''
        while True:
            gp = (i - 3) // 4
            # Grandparent index is valid
            if gp >= 0:
                h_i = self._storage[i]      # Current value
                h_gp = self._storage[gp]    # Grandparent value
                # Grandparent value is less than current value
                if h_i[self._priority_key] > h_gp[self._priority_key]:
                    # Swap current value with grandparent value
                    self._storage[i], self._storage[gp] = h_gp.clone(), h_i.clone()
                    i = gp
                else: break
            else: break

    def _push_up(self, i: int):
        '''
        Pushes the value at index `i` up the heap to maintain the min-max heap property.
        Called when a new value is pushed or an existing value is updated.
        Determines whether the current index is on a min or max level and calls the appropriate push-up function.
        '''
        # Current index is not the root
        if i > 0:
            p = (i - 1) // 2
            h_p = self._storage[p]
            h_i = self._storage[i]

            # Current index is on a min level
            if floor(log2(i + 1)) % 2 == 0: 
                # Current value is greater than parent value
                if h_i[self._priority_key] > h_p[self._priority_key]:
                    # Swap current value with parent value
                    self._storage[i], self._storage[p] = h_p.clone(), h_i.clone()
                    # Push up the parent value to maintain min-heap property
                    self._push_up_max(p)
                else:
                    self._push_up_min(i)
            # Current index is on a max level
            else:
                # Current value is less than parent value
                if h_i[self._priority_key] < h_p[self._priority_key]:
                    self._storage[i], self._storage[p] = h_p.clone(), h_i.clone()
                    self._push_up_min(p)
                else:
                    self._push_up_max(i)
                
    def push(self, value: TensorDict):
        '''
        Push a new value into the min-max heap. If the heap is full, it replaces the minimum value
        
        Parameters
        ----------
        value : TensorDict
            The value to push into the heap.
        '''
        # Push the value up the heap if there is space
        if self._count < self._capacity:
            self._storage[self._count] = value.clone()
            self._push_up(self._count)
            self._count += 1
        else:
            # Replace the minimum value if the new value has a higher priority
            if value[self._priority_key] > self._storage[0][self._priority_key]:
                self._storage[0] = value.clone()
                self._push_down(0)

    def pop(self, i: int) -> TensorDict:
        '''
        Pops the value at index `i` from the min-max heap.

        Parameters
        ----------
        i : int
            The index of the value to pop. Must be 0 for minimum or 1 or 2 for maximum.

        Returns
        -------
        val : TensorDict
            The value at index `i` in the heap.
        '''
        if self._count == 0:
            raise IndexError("Attempted to pop from an empty heap.")
        value = self._storage[i].clone()
        self._storage[i] = self._storage[self._count-1].clone()
        self._storage[self._count-1] = self._proto_experience.clone()
        self._count -= 1
        self._push_down(i)
        return value

    def pop_min(self) -> TensorDict:
        '''
        Pops the minimum n values from the min-max heap.

        Parameters
        ----------
        n : int, optional
            The number of minimum values to pop. Defaults to 1.

        Returns
        -------
        min_vals : TensorDict
            The n minimum values in the heap.
        '''
        return self.pop(0)
    
    def pop_max(self) -> TensorDict:
        '''
        Pops the maximum n values from the min-max heap.

        Parameters
        ----------
        n : int, optional
            The number of maximum values to pop. Defaults to 1.

        Returns
        -------
        max_vals : TensorDict
            The n maximum values in the heap.
        '''
        i_max = int(self._count == 2) + \
                int(self._count > 2) * max([1, 2], key=lambda i: self._storage[i][self._priority_key])
        return self.pop(i_max)
    
    def peek_min(self) -> TensorDict:
        '''
        Peeks at the minimum value in the min-max heap without removing it.

        Returns
        -------
        min_val : TensorDict
            The minimum value in the heap, which is the root of the heap.
        '''
        if self._count == 0:
            raise IndexError("Attempted to peek at an empty heap.")
        return self._storage[0]
    
    def peek_max(self) -> TensorDict:
        '''
        Peeks at the maximum value in the min-max heap without removing it.

        Returns
        -------
        max_val : TensorDict
            The maximum value in the heap, which is either at index 1 or 2.
        '''
        if self._count == 0:
            raise IndexError("Attempted to peek at an empty heap.")
        i_max = int(self._count == 2) + \
                int(self._count > 2) * max([1, 2], key=lambda i: self._storage[i][self._priority_key])
        return self._storage[i_max]
