
import torch
from torch.nn import Module
from tensordict.nn import TensorDictModuleBase

def disable_grads(*modules: Module):
    '''
    Disable gradients for the given modules.

    Parameters
    ----------
    *modules : nn.Module
        The modules for which to disable gradients.
    '''
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

def enable_grads(*modules: Module):
    '''
    Enable gradients for the given modules.

    Parameters
    ----------
    *modules : nn.Module
        The modules for which to enable gradients.
    '''
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True

def polyak_update(target: Module, source: Module, tau: float = 0.005):
    '''
    Optimized functional for performing Polyak updates (soft updates).
    Usually used for updating target network parameters.
    
    *target = polyak \* source + (1 - polyak) \* target*

    Parameters
    ----------
    target : nn.Module
        Target network
    source : nn.Module
        Source network
    tau : float, optional
        The soft update coefficient (default is 0.005).
    '''
    with torch.no_grad():
        for targ_param, param in zip(target.parameters(), source.parameters()):
            targ_param.data.mul_(1 - tau)
            targ_param.data.add_(param.data, alpha=tau)

def prefix_keys(td_module: TensorDictModuleBase, prefix: str):
    '''
    Modify the input and output keys of a TensorDictModule in-place with the given prefix.

    *e.g., A key "val" prefixed with "targ_" would become "targ_val".*

    Parameters
    ----------
    td_module : TensorDictModuleBase
        The TensorDictModule whose keys will be modified.
    prefix : str
        The prefix to add to the keys.
    '''
    for i, key in enumerate(td_module.in_keys):
        if not key.startswith(prefix):
            td_module.in_keys[i] = f"{prefix}{key}"

    for i, key in enumerate(td_module.out_keys):
        if not key.startswith(prefix):
            td_module.out_keys_source[i] = f"{prefix}{key}"
