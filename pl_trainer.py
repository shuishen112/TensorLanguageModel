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
from classification_models.lightCNN import litCNN
from classification_models.lightWordvec import litWord2vec
from classification_models.lightLSTM import litLSTM
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
data_module = ClassificationDataModule(data_name=args.data_name)

wandb_logger = WandbLogger(
    project="TextClasfication",
    config=args,
    name=f"{args.cell}_{args.data_name}",
)
# wandb.define_metric("val_epoch_acc", summary="max")

if args.cell in ["RNN", "MRNN", "MIRNN", "RACs", "Second"]:
    model = RNN_Model_for_classfication(
        vocab_size=len(data_module.vocab),
        embed_size=args.embed_size,
        hidden_dim=args.hidden_size,
        output_size=args.output_size,
    )
elif args.cell in ["TinyTNLM"]:
    model = TN_model_for_classfication(
        rank=args.rank,
        vocab_size=len(data_module.vocab),
        output_size=args.output_size,
        dropout=args.dropout,
        activation=args.activation,
    )
elif args.cell in ["CNN"]:
    model = litCNN(
        len(data_module.vocab), e_dim=args.embed_size, h_dim=args.hidden_size, o_dim=args.output_size, embedding = data_module.embeddings
    )
elif args.cell in ["word2vec"]:
    model = litWord2vec(
        len(data_module.vocab), e_dim=args.embed_size, o_dim=args.output_size, embedding = data_module.embeddings
    )
elif args.cell in ["LSTM"]:
    model = litLSTM(
        len(data_module.vocab), embedding_dim=args.embed_size, hidden_dim=args.hidden_size, num_classes=args.output_size, lstm_layers = 2, bidirectional = True,dropout=0.5,pad_index = 0, embedding = data_module.embeddings,
    )
else:
    print("there is no cell")

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    dirpath=f"output/{args.cell}_{args.data_name}",
    save_top_k=1,
    filename="sample-{epoch:02d}-{val_acc:.2f}",
    mode="max",
)


if __name__ == '__main__':

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=args.epoch,
        # max_epochs=1,
        accelerator="gpu",
        # limit_train_batches=10,
        callbacks=checkpoint_callback,
    )

    trainer.fit(model, datamodule=data_module)

    result = trainer.test(model, data_module, ckpt_path="best")
    print(result)
