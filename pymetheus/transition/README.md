[**< Back**](../README.md)

---

# Transition Function Module

This module contains implementations of transition functions used by environments or environment models. In the former case, the transition function uses the environment to determine the next state given the current state and action. In the latter case, the transition function uses a dynamics model (either learned or given) to predict the next state given the current state and action.

[transition/base.py](./base.py) contains the abstract `Transition` class, which defines the contract for interaction with transition functions. All concrete implementations must subclass `Transition` to ensure this contract is maintained.

---

## Implementations

### 


