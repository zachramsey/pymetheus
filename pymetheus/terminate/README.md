[**< Back**](../README.md)

---

# Termination Function (Done) Module

This module contains implementations of termination functions used by environments or environment models. In the former case, the termination function uses the environment to determine when an episode is terminal given the current state and action. In the latter case, the termination function uses a model (either learned or given) to predict whether the episode is terminal given the current state and action. When the episode is terminal, the termination function sets the done flag, and the environment is automatically reset at the following step.

[done/base.py](./base.py) contains the abstract `Done` class, which defines the contract for interaction with termination functions. All concrete implementations must subclass `Done` to ensure this contract is maintained.

---

## Implementations

### 


