import argparse
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from classification_models.lightCNN import litCNN
from classification_models.lightSimpleRNN import litSimpleRNN
from classification_models.lightTT import litTNLM
from config import args
from utils.cls_data_process import ClassificationDataModule


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.gpus > 0:
    #     torch.cuda.manual_seed_all(args.seed)


set_seed(args)

# generate data batch and iterator
data_module = ClassificationDataModule(data_name="sst2")

wandb_logger = WandbLogger(project="ICLR", config=args)
wandb.define_metric("val_epoch_acc", summary="max")
# model = litCNN(
#     len(vocab), e_dim=args.embed_size, h_dim=args.hidden_size, o_dim=args.output_size
# )

# model = litSimpleRNN(
#     vocab_size=len(data_module.vocab),
#     embed_size=args.embed_size,
#     hidden_dim=args.hidden_size,
#     output_size=args.output_size,
# )
model = litTNLM(
    rank=args.rank,
    vocab_size=len(data_module.vocab),
    output_size=args.output_size,
    dropout=args.dropout,
    activation=args.activation,
)

trainer = pl.Trainer(logger=wandb_logger, max_epochs=args.epoch, accelerator="gpu")
trainer.fit(model, datamodule=data_module)
