from pymetheus.buffer import FIFO
import random
from tensordict import TensorDict   # type: ignore
import torch
import unittest


class TestUniform(unittest.TestCase):
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
        self.buffer = FIFO(
            self.proto_experience,
            self.capacity,
            self.batch_size,
            priority_key=self.priority_key,
            device=self.device)

        self.init_priorities = [8, 10, 11, 13, 16, 21, 31, 31, 41, 46, 51, 71]
        for i, priority in enumerate(self.init_priorities):
            experience = self._make_item(priority, i)
            self.buffer.add(experience)

    def _make_item(self, priority, step):
        experience = self.proto_experience.clone()
        experience["step"] = torch.tensor([step])
        experience[self.priority_key] = torch.tensor([priority])
        return experience

    def _validate_structure(self):
        '''  '''
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
        self.assertEqual(self.buffer._size, len(self.init_priorities[1:]))
        self.assertEqual(self.buffer._next, len(self.init_priorities[1:]))
        self.assertEqual(len(self.buffer), len(self.init_priorities[1:]))
        self.assertEqual(len(self.buffer._priorities), self.capacity)
        self.assertEqual(self.buffer.total, sum(self.init_priorities[1:]))
        self.assertEqual(self.buffer.max, max(self.init_priorities[1:]))
        self.assertTrue(*self._validate_structure())

    def test_clear(self):
        self.buffer.clear()
        self.assertTrue(
            all(torch.equal(self.buffer.proto_experience[key],
                            self.proto_experience[key])
                for key in self.proto_experience.keys()))
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(len(self.buffer._priorities), self.capacity)
        self.assertEqual(self.buffer.total, 0.0)
        self.assertEqual(self.buffer.max, 1.0)

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
        pass

    def test_update(self):
        for _ in range(100):
            idcs, old_priorities, _ = self.buffer.sample(10)
            new_priorities = [abs(priority + random.randint(-20, 20))
                              for priority in old_priorities]
            old_total = self.buffer.total
            self.buffer.update(idcs, new_priorities)
            self.assertEqual(len(self.buffer), len(self.init_priorities) - 1)
            unique = {}
            for idx, new_priority, old_priority in zip(idcs, new_priorities,
                                                       old_priorities):
                if idx not in unique or new_priority > unique[idx][0]:
                    unique[idx] = (new_priority, old_priority)
            diff = sum(new - old for (new, old) in unique.values())
            self.assertEqual(self.buffer.total, old_total + diff)
            self.assertEqual(self.buffer.max, max(self.buffer.max,
                                                  *new_priorities))
            self.assertTrue(*self._validate_structure())
