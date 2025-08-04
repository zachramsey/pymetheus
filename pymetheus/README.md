
# Metheus Modules

[`buffer`](buffer/README.md) — *Stores collected experiences for immediate or later learning.*
```
├── base.py         | Abstract base class for implementations of experience buffers.
└── 
```

[`distribution`](distribution/README.md) — *`TODOC`*
```
├── 
└── 
```

[`environment`](environment/README.md) — *The setting in which agents take actions and receive feedback.*
```
├── base.py         | Abstract base class for implementations of environments.
└── 
```

[`estimate`](estimate/README.md) — *Updates predictions of expected cumulative rewards.*
```
├── base.py         | Abstract base class for implementations of value estimators.
└── 
```

[`modules`](done/README.md) — *Additional modules used to augment agent learning.*
```
├── 
└── 
```

[`policy`](policy/README.md) — *Determines the action to take that maximizes returns given the current state.*
```
├── base.py         | Abstract base class for implementations of policy functions.
└── 
```

[`reset`](reset/README.md) — *Resets the environment after an episode is flagged as terminal.*
```
├── base.py         | Abstract base class for implementations of environment reset functions.
└── 
```

[`reward`](reward/README.md) — *Computes the reward for transitioning between states when an action is taken.*
```
├── base.py         | Abstract base class for implementations of reward functions.
└── 
```

[`terminate`](done/README.md) — *Sets the environment done flag when an episode is terminal.*
```
├── base.py         | Abstract base class for implementations of environment termination logic.
└── 
```

[`transition`](transition/README.md) — *Determines the next state given the current state and action.*
```
├── base.py         | Abstract base class for implementations of environment transition functions.
└── 
```

[`utils`](utils/README.md) — *Utilities for data management, logging, visualization, etc.*
```
├── 
└── 
```

[`value`](value/README.md) — *Estimates the expected attainable returns given a state or state-action pair.*
```
├── base.py         | Abstract base class for implementations of value functions.
└── 
```
