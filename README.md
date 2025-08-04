# Metheus

***Ignite Deep Reinforcement Learning with PyTorch***

Metheus is a package designed to accelerate research, development, and deployment of deep reinforcement learning (DRL) pipelines. The primary objective of this project is to seemlessly extend the deep learning capabilities of PyTorch into the reinforcement learning setting without compromising usability.

---

### Structure

Metheus is a collection of standardized modules, most of which include at least three files: `__init__.py`, `base.py`, and `README.md`. The former and latter are standard to treat the directory as a module and provide documentation. `base.py` provides the abstract base class from which all remaining files in the module are derived.

***Note**: `base.py` is omitted from `mixins` and `utils` due to their heterogeneous implementations.*

Following from the standard reinforcement learning schema, Metheus is built around the cyclical exchange between an agent and its environment. An environment is composed of four 

See [prometheus/README.md](prometheus/README.md) for further details.

---