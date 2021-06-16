from functools import wraps
from typing import Callable

import torch
from torch import Tensor

from .context_managers import RandContext


def cached(func: Callable[..., Tensor]):
    """
    A decorator that takes a model call function into a cached compatible version.
    :param func: A function that calls the model and return representation tensor.
    :return: A function that returns 1) representation leaf tensors for cache construction, 2) a closure function for
    the 2nd forward and the cached backward. Call 2) with 1) as argument after calling backward on the loss Tensor.
    """
    @wraps(func)
    def cache_func(*args, **kwargs):
        rnd_state = RandContext()
        with torch.no_grad():
            reps_no_grad = func(*args, **kwargs)
        leaf_reps = reps_no_grad.detach().requires_grad_()

        @wraps(func)
        def forward_backward_func(cache_reps: Tensor):
            with rnd_state:
                reps = func(*args, **kwargs)
            surrogate = torch.dot(reps.flatten(), cache_reps.grad.flatten())
            surrogate.backward()
        return leaf_reps, forward_backward_func
    return cache_func


def _cat_tensor_list(xx):
    if isinstance(xx, list) and len(xx) > 0 and all(isinstance(x, Tensor) for x in xx):
        return torch.cat(xx)
    else:
        return xx


def cat_input_tensor(func: Callable[..., Tensor]):
    """
    A decorator that concatenates positional and keyword arguments of type List[Tensor] into a single Tensor
    on the 0 dimension. This can come in handy dealing with results of representation tensors from multiple
    cached forward.
    :param func: A loss function
    :return: Decorated loss function for cached results.
    """
    @wraps(func)
    def cat_f(*args, **kwargs):
        args_cat = [_cat_tensor_list(x) for x in args]
        kwargs_cat = dict((k, _cat_tensor_list(v)) for k, v in kwargs.values())
        return func(*args_cat, **kwargs_cat)
    return cat_f
