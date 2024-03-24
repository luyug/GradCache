"""
PyTorch Lightning Example of using Grad Cache, tested on PyTorch Lightning version '2.2.0.post0' with Multi-GPUs and Mix-Precision (fp-16).
Required to install Pytorch Metric Learning as well for contrastive loss calculation.
"""

import os
import argparse
import torch
import lightning as pl
from contextlib import nullcontext
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from pytorch_metric_learning.utils import distributed as pml_dist
from pytorch_metric_learning.losses import SupConLoss

from grad_cache.pytorch_lightning.pl_gradcache import PLGradCache


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        self.params = params

    def __len__(self):
        return self.params.data_size

    def __getitem__(self, idx):
        # Generate random float inputs with shape [2, input_dim] for contrastive learning
        input_data = torch.randn(2, self.params.input_dim)
        # Generate a random integer label for binary classification (0 or 1), replicate it to have shape [2]
        label = torch.randint(0, 2, (1,), dtype=torch.long)
        label = torch.tensor([label, label], dtype=torch.long)
        return input_data, label


class SimpleLitModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.loss = SupConLoss(temperature=params.temperature)
        if params.gpus > 1:
            self.loss = pml_dist.DistributedLossWrapper(self.loss)
        self.automatic_optimization = (not self.params.use_gc) # needed when use_gc is on
        self.fp16 = (self.params.precision == 16)
        self.linear = torch.nn.Linear(params.input_dim, params.embed_dim) # our simple model

    def init_gc(self, scaler, ddp_module):
        """Sets up the required components of GradCache. This method is called after the model is initialized."""
        assert self.params.use_gc
        if self.fp16 and self.params.use_gc:
            # pytorch lightning autocast wraps everything in it
            # it needs to be disabled in gradcache because we do forward twice, and one with no grad
            # then we do autocast manually in gradcache when we need to
            # original post: https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/3
            # pl source code: your_venv_name/lib/python3.8/site-packages/lightning/pytorch/plugins/precision/amp.py::forward_context
            self.trainer.strategy.precision_plugin.forward_context = nullcontext

        print(f"*** initializing gradcache with ddp_module={type(ddp_module)}, minibatch_size={self.params.gc_minibatch_size}")
        self.gc = PLGradCache(
            models=[ddp_module],
            chunk_sizes=self.params.gc_minibatch_size,
            loss_fn=self.calculate_loss,
            fp16=self.fp16,
            scaler=(scaler if self.fp16 else None), # needed when using automatic_optimization is off and fp16 is on
            backward_fn=self.manual_backward, # needed when automatic_optimization is off
        )

    def train_dataloader(self):
        train_dataset = RandomDataset(params)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            drop_last=True,
        )
        return train_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def calculate_loss(self, embeddings, labels):
        # embeddings shape [batch_size, 2, embed_dim]
        # labels shape [batch_size, 2]
        embeddings = embeddings.flatten(0, 1)
        labels = labels.flatten()
        return self.loss(embeddings, labels)
    
    def forward(self, inputs): # needed for grad cache
        return self.linear(inputs)
    
    def on_train_start(self): # initialize grad cache here
        if self.params.use_gc:
            self.init_gc(self.trainer.scaler, self.trainer.strategy.model)
            # self.init_gc(self.trainer.scaler, self.trainer.lightning_module) # we can use this if nccl strategy is available

    def training_step(self, batch, batch_idx):
        # inputs shape [batch_size, 2, input_dim]
        # labels shape [batch_size, 2]
        inputs, labels = batch
        if self.params.use_gc:
            assert self.gc is not None
            optimizer = self.optimizers()
            optimizer.zero_grad()
            loss = self.gc(
                inputs,
                no_sync_except_last=(self.params.gpus > 1),
                labels=labels.flatten(),
            )
            loss /= max(1, self.params.gpus) # needed when automatic_optimization is off
            log_loss = loss
            optimizer.step()
        else:
            outputs = self.linear(inputs)
            loss = self.calculate_loss(outputs, labels)
            log_loss = loss / max(1, self.params.gpus)
        self.log(
            "train_loss",
            log_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=self.params.use_gc, # needed when automatic_optimization is off
        )
        print(f"batch_idx={batch_idx}, loss={loss}")
        return loss


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="torch distributed backend (Default: nccl), use 'gloo' if nccl doesn't work")
    parser.add_argument("--project_name", type=str, default="debug_gradcache")

    # training params
    parser.add_argument("--data_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)

    # model hyperparams
    parser.add_argument("--input_dim", type=int, default=784)
    parser.add_argument("--embed_dim", type=int, default=512)

    # grad cache params
    parser.add_argument("--use_gc", action="store_true", default=False, help="whether to use grad cache")
    parser.add_argument("--gc_minibatch_size", type=int, default=2, help="mini batch size of grad cache, must be provided if use_gc is on")

    return parser


def main(params):
    # set random seeds reproduceability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set different random seeds for each worker
    pl.seed_everything(seed=params.random_seed, workers=True)

    # weirdness with HuggingFace tokenizer when processing things in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy("file_system")

    # load model
    model = SimpleLitModel(params)

    # load trainer
    experiment_id = f"gpus-{params.gpus}_precision-{params.precision}"
    if params.use_gc:
        experiment_id += "_gc"
    experiment_id += "_pl"
    wandb_logger = WandbLogger(
        project=params.project_name,
        name=experiment_id,
    )
    ddp = DDPStrategy(process_group_backend=params.ddp_backend)
    trainer = pl.Trainer(
        accelerator="gpu" if params.gpus > 0 else "cpu",
        strategy=ddp if params.gpus > 1 else "auto",
        devices=params.gpus if params.gpus > 0 else "auto",
        precision=params.precision,
        logger=wandb_logger,
        max_epochs=params.epochs,
        log_every_n_steps=1,
    )
    trainer.fit(model)


if __name__ == "__main__":
    params = get_argument_parser().parse_args()
    main(params)
