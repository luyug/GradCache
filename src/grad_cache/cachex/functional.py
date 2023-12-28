from typing import Iterable, Any, Callable
from functools import partial

import jax
import jax.numpy as jnp

from .tree_utils import tree_unchunk

Array = jax.Array


def grad_with_cache(f, **grad_kwargs):
    def cache_f(params, cache, *args, **kwargs):
        return jnp.sum(f(params, *args, **kwargs) * cache)
    return jax.grad(cache_f, **grad_kwargs)


def encode_scan_fn(f, carry, x):
    return carry, f(**x)


def cache_grad_scan_fn(f, params, acc, x):
    cached_grad, kwargs = x

    def fwd_fn(w):
        return f(params=w, **kwargs)

    chunk_grad = grad_with_cache(fwd_fn)(params, cached_grad)
    acc = jax.tree_multimap(lambda u, v: u + v, acc, chunk_grad)
    return acc, None


def chunk_encode(encode_fn):
    def f(**xx):
        _, hh = jax.lax.scan(partial(encode_scan_fn, encode_fn), 0, xx)
        return hh
    return f


def cache_grad(encode_fn):
    def f(params, grad_accumulator, cached_grad, **xx):
        grads, _ = jax.lax.scan(
            partial(cache_grad_scan_fn, encode_fn, params), grad_accumulator, [cached_grad, xx]
        )
        return grads
    return f


def unchunk_args(axis: int = 0, argnums: Iterable[int] = ()):
    def decorator_unchunk(f):
        def g(*args, **kwargs):
            new_args = list(args)
            for i in argnums:
                new_args[i] = tree_unchunk(args[i], axis)
            return f(*new_args, **kwargs)

        return g

    return decorator_unchunk

def grad_cached(
    f: Callable[..., Array],
    policy: Callable[..., bool] = jax.checkpoint_policies.nothing_saveable,
    prevent_cse: bool = True
):
    """
    Single-decorator version of grad cache that uses XLA to infer backward pass.
    
    The forward pass is manually split into chunks and performed sequentially with lax.scan.
    We rely on XLA to infer the backward pass and run it in a similar fashion.
    
    Args:
        f: Function to be differentiated. It should take in a single argument and return a jax array of representations.
        policy: The sub-batch rematerialization policy.
        prevent_cse: Whether to prevent common subexpression elimination.
        
    Returns:
    Decorated gradient cached `f` that expects input to have an extra leading sub-batch dimension, potentially produced by `tree_chunk`.
    
    A example of usage on a apply function that takes multiple arguments:
    
    >>> @cachex.grad_cached
    ... def fwd(params, batch):
    ...     return apply(params, **batch)
    
    >>> src = cachex.tree_chunk(src, 8)
    >>> tgt = cachex.tree_chunk(tgt, 8)

    >>> def compute_loss(params, src, tgt):
    ...     h_src = fwd(params, src)
    ...     h_tgt = fwd(params, tgt)
    ...     return loss(h_src, h_tgt)
    
    >>> grads = jax.grad(compute_loss)(params, src, tgt)
    
    Here the `compute_loss` function can typically be dropped into a larger training step function.
    """
    def cached_f(params, batch):
        def scan_f(_, sub_batch):
            return None, f(params, sub_batch)
        _, reps = jax.lax.scan(jax.checkpoint(scan_f, policy=policy, prevent_cse=prevent_cse), None, batch)
        return jnp.reshape(reps, (-1,) + reps.shape[2:])
    return cached_f