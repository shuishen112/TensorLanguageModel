import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import random

import numpy as np

import torch
from pytorch_lightning.loggers import WandbLogger
import wandb
from classification_models.lightRNN import RNN_Model_for_classfication
from classification_models.lightTT import TN_model_for_classfication
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

wandb_logger = WandbLogger(
    project="ICLR", config=args, name=f"{args.cell}_{args.data_name}"
)
wandb.define_metric("val_epoch_acc", summary="max")
# model = litCNN(
#     len(vocab), e_dim=args.embed_size, h_dim=args.hidden_size, o_dim=args.output_size
# )

# model = RNN_Model_for_classfication(
#     vocab_size=len(data_module.vocab),
#     embed_size=args.embed_size,
#     hidden_dim=args.hidden_size,
#     output_size=args.output_size,
# )
model = TN_model_for_classfication(
    rank=args.rank,
    vocab_size=len(data_module.vocab),
    output_size=args.output_size,
    dropout=args.dropout,
    activation=args.activation,
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    dirpath=f"output/{args.cell}_{args.data_name}",
    save_top_k=2,
    filename="sample-{epoch:02d}-{val_acc:.2f}",
    mode="max",
)
trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=20,
    accelerator="gpu",
    # limit_train_batches=10,
    callbacks=checkpoint_callback,
)
trainer.fit(model, datamodule=data_module)

result = trainer.test(model, data_module, ckpt_path="best")
print(result)
