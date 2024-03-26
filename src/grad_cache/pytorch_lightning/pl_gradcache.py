from contextlib import nullcontext
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast

from ..grad_cache import GradCache, RandContext

class PLGradCache(GradCache):
    """
    Gradient Cache class with PyTorch Lightning Support.
    Implements input chunking, first graph-less forward pass, Gradient Cache creation, second forward & backward gradient computation.
    Optimizer step is not included. Native torch automatic mixed precision is supported.
    Gradient unscaling and scaler update are handled internally.
    """

    def __init__(
        self,
        models: List[nn.Module],
        chunk_sizes: Union[int, List[int]],
        loss_fn: Callable[..., Tensor],
        split_input_fn: Callable[[Any, int], Any] = None,
        get_rep_fn: Callable[..., Tensor] = None,
        fp16: bool = False,
        scaler: GradScaler = None,
        backward_fn=None,  # [added]
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        :[added] param backward_fn: The `manual_backward` function of pytorch lightning trainer when automatic_optimization is disabled.
        """
        super().__init__(models, chunk_sizes, loss_fn, split_input_fn, get_rep_fn, fp16, scaler)
        self.backward_fn = backward_fn

    def build_cache(self, *reps: Tensor, **loss_kwargs) -> Union[List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        reps = [r.detach().requires_grad_() for r in reps]
        with autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(*reps, **loss_kwargs)

        self.backward_fn(loss)  # [modified]

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    def forward_backward(
        self,
        model: nn.Module,
        model_inputs,
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False,
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if isinstance(
            model, nn.parallel.DistributedDataParallel
        ):  # [use ddp_model]

            if no_sync_except_last:
                sync_contexts = [
                    model.no_sync for _ in range(len(model_inputs) - 1)
                ] + [nullcontext]
                sync_flags = [True] * (len(model_inputs))  # [added]
            else:
                sync_contexts = [nullcontext for _ in range(len(model_inputs))]
                sync_flags = [False] * (len(model_inputs))  # [added]

            # [modified]
            for x, state, gradient, sync_context, sync_flag in zip(
                model_inputs, random_states, cached_gradients, sync_contexts, sync_flags
            ):
                with sync_context():
                    with state:
                        y = self.model_call(model, x)
                    reps = self.get_reps(y)
                    surrogate = torch.dot(reps.flatten(), gradient.flatten())
                    if sync_flag:
                        model.require_backward_grad_sync = True
                    if self.fp16:  # [added]
                        self.scaler._enabled = False
                        self.backward_fn(surrogate)
                        self.scaler._enabled = True
                    else:
                        self.backward_fn(surrogate)  # [modified]
        else:  # [use base model (i.e. SimpleLitModel)]

            # [remove no_sync_except_last: pytorch lightning would handle gradient sync automatically]
            for x, state, gradient in zip(
                model_inputs, random_states, cached_gradients
            ):
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)
                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                if self.fp16:  # [added]
                    self.scaler._enabled = False
                    self.backward_fn(surrogate)
                    self.scaler._enabled = True
                else:
                    self.backward_fn(surrogate)  # [added]

    def cache_step(
        self, *model_inputs, no_sync_except_last: bool = False, **loss_kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: A tuple of the current's loss and the model's representation.
        """
        all_reps = []
        all_rnd_states = []

        # [removed: we check it in forward_backward(.)]
        # if no_sync_except_last:
        #     assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
        #         'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
        #         'proper initializations.'

        model_inputs = [
            self.split_inputs(x, chunk_size)
            for x, chunk_size in zip(model_inputs, self.chunk_sizes)
        ]

        for model, x in zip(self.models, model_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        # all_reps: len(self.models) x [batch_size, 2, embed_dim]
        # cache: len(self.models) x gc_minibatch x [(batch_size / gc_minibatch, 2, embed_dim]

        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for model, x, model_cache, rnd_states in zip(
            self.models, model_inputs, cache, all_rnd_states
        ):
            self.forward_backward(
                model,
                x,
                model_cache,
                rnd_states,
                no_sync_except_last=no_sync_except_last,
            )

        return loss
