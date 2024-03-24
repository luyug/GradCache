# PL_GradCache

This is an experimental folder to provide example of using Grad Cache with PyTorch Lightning (pl), tested on pl version '2.2.0.post0' with Multi-GPUs and Mix-Precision (fp-16). Pytorch Metric Learning is required to install as well for contrastive loss calculation.

- [Wandb Logging Experiments for Sanity Test](https://api.wandb.ai/links/xyznlp/nmf8d551)

### Installation

After GradCache is installed, do

```
cd GradCache/src/grad_cache/pytorch_lightning
python -m venv plgc
. ./plgc/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

### Reproducing Wandb Experiments

```
# 1-gpu
python pl_example.py --gpus 1 --batch_size 16
# 2-gpus
python pl_example.py --gpus 2 --batch_size 8
# 1-gpu, gradcache
python pl_example.py --gpus 1 --batch_size 16 --use_gc --gc_minibatch_size 2
# 2-gpus, gradcache
python pl_example.py --gpus 2 --batch_size 8 --use_gc --gc_minibatch_size 2
```

Optionally, do mix-precision training with `--precision 16`, run different ddp_backend with `--ddp_backend {gloo/nccl/etc.}`

### Example

Run `python pl_example.py` with the following flags.

* `--use_gc` activates GradCache.
* `--gc_minibatch_size {minibatch_size}` defines the batch size that each GPU needs to hold its memory into. If we specify `--gpus 2 --batch_size 8 --gc_minibatch 2`, for example, the model would be trained with batch size 8 * 2 = 16, the trainer would split each batch on each GPU (8 data samples) into 4 chunks of mini batches (2 data samples per mini batch). Set this to 1 gives the minimal possible gpu memory usage.

### Summary

- Add `pl_gradcache.py` as customized GradCache on PyTorch Lightning.
- Use manual backward in gradcache by calling `lightning_trainer.manual_backward(loss)` instead of using `loss.backward()` (this requires changing gradcache).
- Set gradcache `no_sync_except_last=True` in multi-GPU case.

### Changes to the original GradCache

#### File Change
- `pl_gradcache.py` is the GradCache we will run on PyTorch Lightning (pl) with Distributed Data Parallel (ddp).

#### Change in Optimization
- In pt ddp setting, we need to first set `lightning_trainer.automatic_optimization=False` for us to customize calling backward.
- See [the pl optimization doc](https://lightning.ai/docs/pytorch/stable/common/optimization.html) for implementation details, make sure that we are calling `self.optimizers()` instead of creating one by ourselves: if we do `self.optimizer = optimizer` in `self.configure_optimizers()`, this is not correct as it initializes a base optimizer in pt, but `self.optimizers()` is a wrapper for that. The base optimizer does not have the correct access to ddp and logging.
- Then, replace all `loss.backward()` in GradCache with `lightning_trainer.manual_backward(loss)`.

#### Change in GradCache
- Set `no_sync_except_last=True` in Multi-GPU case to avoid unnecessary gradient reduction in the last step of gradcache.

#### If you want to run GradCache in PyTorch Lightning with Multi-GPUs
- In short, you are good to go by not worrying about this part. But here are the key changes in the original gradcache that are necessary for this to work.
- we have two options to use gradcache in pl ddp setting and call `self.init_gc(scaler, gc_model)`.
- We can set `gc_model=pytorch lightning trainer`.
  - PyTorch Lightning would then wrap the base model (transformer) by their implementation of DDP. 
  - In this case, just set `no_sync_except_last = False`, because lightning will handle gradient sync before `optimizer.step()`.
  - Set `no_sync_except_last = True` in this case does not work as the base model in gradcache is the transformer, which causes gradcache assertion and `model.no_sync` not available error.
  - Or, we can just change gradcache instead (remove assert DDP and `model.no_sync`). 
  - The only downside of this approach is that the training may take a little longer, because gradient sync is done on the full batch size (expected by pytorch lightning) instead of the last minibatch (expected by gradcache). But based on some sanity runs, it is ok (less than a 10% runtime increase).
- We can set `gc_model=pytorch lightning trainer.strategy.model`, i.e. the wrapped base model by PyTorch DDP.
  - This is tricky as PyTorch Lightning uses a parameter `require_backward_grad_sync` to determine whether gradients would be synced across GPUs.
  - Firstly, Pytorch Lightning overrides the PyTorch DDP by their own implementation and set `require_backward_grad_sync=False` before each training step (when `automatic optimization=False`). Then, it is set it to True **after** each training step. 
  - The issue here is that gradcache needs the gradient to be synced in the last backward step, which happens inside the training step hook of pytorch lightning. Thus, what we can only do is to set this variable manually before the last backward step in gradcache - we cannot set it outside of gradcache either, because the first backward of gradcache to do gradient checkpointing should NOT sync gradient (this is the point of gradcache essentially). 
  - Thus, we do `model.require_backward_grad_sync=True` at the very end of gradcache - before the backward of the last minibatch surrogate.
  - The advantage of this is that we can do `no_sync_except_last` as what gradcache hopes us to do (no runtime increase). The downside is that we need to modify gradcache in a very hacky way. This is the default setup.
