[**< Back**](../README.md)

---

# Reward Function Module

This module contains implementations of reward functions used by environments or environment models. In the former case, the reward function uses the environment and calculate the reward for an agent taking an action in the current state and transitioning into the next state. In the latter case, the reward function uses a reward model (either learned or given) to predict the reward for an agent taking an action in the current state and transitioning into the next state.

[reward/base.py](./base.py) contains the abstract `Reward` class, which defines the contract for interaction with reward functions. All concrete implementations must subclass `Reward` to ensure this contract is maintained.

---

## Implementations

### 


