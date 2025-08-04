[**< Back**](../README.md)

---

# Environment Module

This module contains implementations of environments for direct (model-free) reinforcement learning and environment models for indirect (model-based) reinforcement learning. In the former case, an environment provides the real next-state, reward, and done flag given the last state and action taken in that state. In the latter case, an environment model (either learned or given) predicts the next-state, reward, and done flag given some state and action taken in that state.

[environment/base.py](./base.py) contains the abstract `Environment` class, which defines the contract for interaction with environments and environment models. All concrete implementations must subclass `Environment` to ensure this contract is maintained.

---

## Implementations

### 


