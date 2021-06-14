

# Gradient Cache
Gradient Cache is a simple technique for unlimitedly scaling contrastive learning batch far beyond GPU memory constraint. This means training that used to take heavy hardware, e.g. 8 V100 GPU, can be done on a single GPU. In addition, Gradient Cache allow users to replace big RAM GPU with much more cost efficient high FLOP low RAM cards.

This repo holds a generic Pytorch implementation of Gradient Cache described in our paper [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup
](https://arxiv.org/abs/2101.06983).
```
@inproceedings{gao2021scaling,
     title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
     author={Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan},
     booktitle ={Proceedings of the 6th Workshop on Representation Learning for NLP},
     year={2021},
}
```

## Installation
The package depends only on `pytorch>=1.6`.  To install, clone this repo and run pip.
```
git clone https://github.com/luyug/GradCache
cd GradCache
pip install .
```
For development,
```
pip install --editable .
```

# Usage
Gradient caching functionalities are implemented in `GradCache` class. The class's `__init__` method has several functional parameters `*_fn` for easy adjust of model behaviors. Alternatively you can also sub-class GradCache.
```
grad_cache.GradCache(  
  models: List[nn.Module],  
  chunk_sizes: Union[int, List[int]],  
  loss_fn: Callable[..., Tensor],  
  split_input_fn: Callable[[Any, int], Any] = None,  
  get_rep_fn: Callable[..., Tensor] = None,  
  fp16: bool = False,  
  scaler: GradScaler = None,  
)
``` 
**models** - A list of encoder models to be updated with with the Gradient Cache.

**chunk_sizes** - An integer indicating chunk size. Or a list of integers of chunk size for each model. This controls for each model the sub-batch size to run forward-backward pass and should be set based on available GPU memory. A value too small will leave the GPU under utilized.

**loss_fn** -  A loss function that takes representation tensors of number equal to number of models in `models` and arbitrary numbers of keyword arguments. It should compute loss based on the input tensors, and in no case modify the input tensors' relations in the autograd graph, which are later relied upon to create the gradient cache.

**split_input_fn** - An optional function that split generic model input into chunks based on defined chunk_sizes. If not provided, this  class will try its best to split the inputs of supported types. See `split_inputs` function.

**get_rep_fn** - An optional function that takes generic model output and return representation tensors. If  not provided, the generic output is assumed to be the representation tensor.

**fp16** - If True, run mixed precision training, which requires scaler to also be set.

**scaler** - A GradScaler object for automatic mixed precision training.

```
cache_step(  
  *model_inputs,  
  no_sync_except_last: bool = False,  
  **loss_kwargs  
)
```
Run a single gradient cache step. Upon function return, updates are computed for each model in `self.models` with gradient populated on the weights, as if the `model_inputs` are run as a huge single batch on sufficiently large hardware.  Calling an GradCache object with `__call__` will also invoke this function.

**model_inputs** - Input to each encoder model. Should be in similar order as `self.models`.

**no_sync_except_last** - If True, under distributed setup, for each model, only trigger gradient reduction across processes for the last sub-batch's forward-backward pass. This could come in handy when dealing with a) large model, and/or b) non trivial number of sub-batches.

**loss_kwargs** - Additional keyword arguments to the loss function `loss_fn`. This is intended to enable flexible loss computation (thanks to dynamic graph in Pytorch) such as reduction, weighting, etc. Potentially, using **loss_kwargs** you can incorporate outputs from those encoder models not tracked by the cache. 

**Retrun** - loss, the current steps loss scaler tensor (detached from the graph).

## Example Usage with Huggingface Transformers
### Learning a Bi-encoder
Say we want to learn a embedding space of labels and text. Consider the following four pairs. (In practice, you will have many more and much longer text entries.)
```
labels = ['fruit', 'meat', 'school', 'company']
texts = [
  'this is an apple', 
  'steak should be cooked medium rare', 
  'cmu is pittsburgh', 
  'apple sells laptop'
]
```

Initialize our encoder models,
```
from transformers import AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoder1 = AutoModel.from_pretrained("bert-base-uncased").cuda()
encoder2 = AutoModel.from_pretrained("bert-base-uncased").cuda()
```
Initialize the GradCache object,
```
from grad_cache import GradCache
from grad_cache.loss import SimpleContrastiveLoss

loss_fn = SimpleContrastiveLoss()
gc = GradCache(
  models=[encoder1, encoder2], 
  chunk_sizes=2, 
  loss_fn=loss_fn, 
  get_rep_fn=lambda v: v.pooler_output
)
```
Here we use the **get_rep_fn** argument to specify a function that takes generic Huggingface model output and return the actual representation tensor. 

Create model input,
```
xx = tokenizer(tt, return_tensors='pt', padding=True)
yy = tokenizer(tt2, return_tensors='pt', padding=True)
```
Run a cache step,
```
gc(xx, yy, reduction='mean')
```
Here we use `reduction='mean'` as a **loss_kwargs** to control loss behavior. With a defined `optimizer`, the full gradient update can be done as,
```
optimizer.zero_grad()
gc(xx, yy, reduction='mean')
optimizer.step()
``` 

### Use Tied Encoder?
This is naturally handled by the (magic of) dynamic graph. You pass shallow copies of the same encoder model to the GradCache init method.
```
tied_encoder = AutoModel.from_pretrained("bert-base-uncased").cuda()
gc = GradCache(
  models=[tied_encoder , tied_encoder], 
  chunk_sizes=2, 
  loss_fn=loss_fn, 
  get_rep_fn=lambda v: v.pooler_output
)
```
### Distributed Training with Multiple GPUs?
We expect cross process communication of representations to be handled by the `loss_fn`. 
```
from grad_cache.loss import DistributedContrastiveLoss
loss_fn_dist = DistributedContrastiveLoss()
```
Properly wrap the the encoder models for gradient reduction,
```
encoder1_ddp = DistributedDataParallel(
	encoder1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
encoder2_ddp = DistributedDataParallel(
	encoder2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
```
You can initialize the cache use the distributed loss and the DDP models,
```
gc = GradCache(
  models=[encoder1_ddp, encoder2_ddp], 
  chunk_sizes=2, 
  loss_fn=loss_fn_dist, 
  get_rep_fn=lambda v: v.pooler_output
)
```
Run a cache step,
```
gc(xx, yy, no_sync_except_last=True, reduction='mean')
```
Set `no_sync_except_last=True` to avoid unnecessary gradient reduction.

## Code Structure
The GradCache class is defined in [grad_cache.py](src/grad_cache/grad_cache.py). The code is under 300 lines including comments. For development, we encourage you to read through it.
