import pytorch_lightning as pl
import torch
from torch import nn
import wandb


def cal_accuracy(probs, target):
    predictions = probs.argmax(dim=1)
    corrects = predictions == target
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy


class Baselighting(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, lengths):
        pass

    # optimizers go into configure_optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # train and validation
    def training_step(self, train_batch, batch_idx):
        text, label, lengths = train_batch
        predictions = self(text, lengths)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("acc", acc)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, val_batch, batch_idx):
        text, label, lengths = val_batch
        predictions = self(text, lengths)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return acc

    def test_step(self, batch, batch_idx):
        text, label, lengths = batch
        predictions = self(text, lengths)

        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return acc

    def training_epoch_end(self, train_step_outputs) -> None:
        all_acc = torch.stack([x["train_acc"] for x in train_step_outputs])
        self.log("train_epoch_acc", torch.mean(all_acc))

    def validation_epoch_end(self, validation_step_outputs):
        all_acc = torch.stack(validation_step_outputs)
        print("val_epoch_acc:", torch.mean(all_acc))
        # wandb.run.summary["best_val_acc"] = torch.mean(all_acc).cpu().numpy()
        self.log("val_epoch_acc", torch.mean(all_acc))
