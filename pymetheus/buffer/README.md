[**< Back**](../README.md)

---

# Experience Replay Buffer Module

This module contains implementations of experience rollout buffers for on-policy learning and experience replay buffers for off-policy learning. In the former case, the buffer only stores experiences from the last full rollout; that is, interactions between the environment and current policy of the agent up until the done flag is set. In the latter case, the buffer stores experiences from all rollouts during training time; that is, any interaction between the environment and any agent policy may be stored indefinitely or until removed according to some eviction policy. When updating the agent, experiences are sampled from the a buffer according to some sampling strategy.

[buffer/base.py](./base.py) contains the abstract `Buffer` class, which defines the contract for interaction with experience buffers. All concrete implementations must subclass `Buffer` to ensure this contract is maintained.

---

## Implementations

### 


