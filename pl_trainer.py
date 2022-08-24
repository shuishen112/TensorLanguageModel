from classification_models.lightCNN import litCNN
from classification_models.lightSimpleRNN import litSimpleRNN
from classification_models.lightTT import litTNLM
from utils.cls_data_process import vocab, loader_train, loader_validation
import wandb
from config import args
import torch
import random
import numpy as np
import argparse
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.gpus > 0:
    #     torch.cuda.manual_seed_all(args.seed)


# generate data batch and iterator

set_seed(args)
wandb_logger = WandbLogger(project="ICLR", config=args)
wandb.define_metric("val_epoch_acc", summary="max")
# model = litCNN(
#     len(vocab), e_dim=args.embed_size, h_dim=args.hidden_size, o_dim=args.output_size
# )

model = litSimpleRNN(
    vocab_size=len(vocab),
    embed_size=args.embed_size,
    hidden_dim=args.hidden_size,
    output_size=args.output_size,
)
# model = litTNLM(
#     rank=args.rank,
#     vocab_size=len(vocab),
#     output_size=args.output_size,
#     dropout=args.dropout,
#     activation=args.activation,
# )

trainer = pl.Trainer(logger=wandb_logger, max_epochs=args.epoch, accelerator="gpu")
trainer.fit(model, loader_train, loader_validation)
