
import random
from pymetheus.buffer import SumHeap
from tensordict import TensorDict   # type: ignore
import torch
import unittest


class TestSumHeap(unittest.TestCase):
    def setUp(self):
        self.proto_experience = TensorDict({
            "obs": torch.zeros([5]),
            "act": torch.zeros([2]),
            "reward": torch.zeros([1]),
            "done": torch.zeros([1]),
            "step": torch.zeros([1])
        })
        self.capacity = 20
        self.batch_size = 10
        self.priority_key = "priority"
        self.device = "cpu"
        self.heapify_period = 1
        self.buffer = SumHeap(
            self.proto_experience,
            self.capacity,
            self.batch_size,
            priority_key=self.priority_key,
            device=self.device,
            heapify_period=self.heapify_period)

        self.init_priorities = [8, 10, 11, 13, 16, 21, 31, 31, 41, 46, 51, 71]
        for i, priority in enumerate(self.init_priorities):
            experience = self._make_item(priority, i)
            self.buffer.add(experience)

    def _make_item(self, priority, step):
        experience = self.proto_experience.clone()
        experience["step"] = torch.tensor([step])
        experience[self.priority_key] = torch.tensor([priority])
        return experience

    def _validate_structure(self) -> tuple[bool, str | None]:
        ''' Validates the sum-tree and min-heap properties of the buffer. '''
        # Validate the sum-tree
        for i in range((len(self.buffer)-1)//2):
            left, right = 2*i+1, 2*i+2
            if self.buffer._tree[i] != (
                    self.buffer._tree[left] + self.buffer._tree[right]):
                return False, (
                    f"Violation of the sum-tree property at index {i}: "
                    f"{self.buffer._tree[i]} != {self.buffer._tree[left]} + "
                    f"{self.buffer._tree[right]}")
        # Validate the min-heap
        for i in range(len(self.buffer)):
            left, right = 2*i+1, 2*i+2
            curr = self.buffer._heap[i]
            if left < len(self.buffer):
                if curr > self.buffer._heap[left]:
                    return False, (
                        f"Violation of the min-heap property at index {i}: "
                        f"{curr} > {self.buffer._heap[left]}")
            if right < len(self.buffer):
                if curr > self.buffer._heap[right]:
                    return False, (
                        f"Violation of the min-heap property at index {i}: "
                        f"{curr} > {self.buffer._heap[right]}")
        return True, None

    def test_init(self):
        self.assertTrue(all(torch.equal(self.buffer.proto_experience[key],
                                        self.proto_experience[key])
                            for key in self.proto_experience.keys()))
        self.assertEqual(self.buffer.capacity, self.capacity)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
        self.assertEqual(self.priority_key, self.priority_key)
        self.assertEqual(self.buffer.device, self.device)
        self.assertEqual(self.buffer.storage.keys(),
                         self.proto_experience.keys())
        self.assertEqual(self.buffer._heapify_period, self.heapify_period)
        self.assertEqual(len(self.buffer._tree), self.capacity-1)
        self.assertEqual(len(self.buffer._heap), self.capacity)
        self.assertEqual(self.buffer._size, len(self.init_priorities[1:]))
        self.assertEqual(len(self.buffer), len(self.init_priorities[1:]))
        self.assertEqual(self.buffer.total, sum(self.init_priorities[1:]))
        self.assertEqual(self.buffer.max, max(self.init_priorities[1:]))
        self.assertEqual(self.buffer._updates, 0)
        self.assertTrue(*self._validate_structure())

    def test_clear(self):
        self.buffer.clear()
        self.assertTrue(all(torch.equal(self.buffer.proto_experience[key],
                                        self.proto_experience[key])
                            for key in self.proto_experience.keys()))
        self.assertEqual(len(self.buffer._tree), self.capacity-1)
        self.assertEqual(len(self.buffer._heap), self.capacity)
        self.assertEqual(self.buffer._size, 0)
        self.assertEqual(self.buffer.total, 0.0)
        self.assertEqual(self.buffer.max, 1.0)
        self.assertEqual(self.buffer._updates, 0)

    def test_push(self):
        priorities = [5, 16, 38, 75]
        for i, priority in enumerate(priorities):
            experience = self._make_item(priority, i + len(self.buffer))
            self.buffer.add(experience)
            self.assertEqual(len(self.buffer),
                             len(self.init_priorities[1:]) + i + 1)
            self.assertEqual(self.buffer.total,
                             sum(self.init_priorities[1:] + priorities[:i+1]))
            self.assertEqual(self.buffer.max,
                             max(self.init_priorities[1:] + priorities[:i+1]))
            self.assertTrue(*self._validate_structure())

    def test_sample(self):
        for priority in [random.randint(1, 100)
                         for _ in range(self.capacity-len(self.buffer))]:
            experience = self._make_item(priority, len(self.buffer))
            self.buffer.add(experience)
        n = 100
        k = 10
        e_dist = [self.buffer._heap[i][0] / self.buffer.total
                  for i in range(len(self.buffer))]
        a_dist = [0] * len(self.buffer)
        for _ in range(n):
            idcs, _, _ = self.buffer.sample(k)
            for idx in idcs:
                a_dist[idx] += 1
        err = [abs(e - (a/(n*k))) for e, a in zip(e_dist, a_dist)]
        err = sum(err) / len(err)
        self.assertTrue(err < 0.05,
                        f"Sample error ({err}) out of spec ({0.05}).")

    def test_update(self):
        for _ in range(100):
            idcs, old_priorities, _ = self.buffer.sample(10)
            new_priorities = [abs(priority + random.randint(-20, 20))
                              for priority in old_priorities]
            old_total = self.buffer.total
            self.buffer.update(idcs, new_priorities)
            self.assertEqual(len(self.buffer), len(self.init_priorities) - 1)
            unique = {}
            for idx, old_priority, new_priority in zip(idcs, old_priorities,
                                                       new_priorities):
                if idx not in unique or new_priority > unique[idx][0]:
                    unique[idx] = (new_priority, old_priority)
            diff = sum(new - old for (new, old) in unique.values())
            self.assertEqual(self.buffer.total, (old_total + diff))
            self.assertEqual(self.buffer.max, max(self.buffer.max,
                                                  *new_priorities))
            self.assertTrue(*self._validate_structure())
