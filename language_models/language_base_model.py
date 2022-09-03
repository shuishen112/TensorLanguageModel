import math
from abc import abstractmethod
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch


class LanguageBaselightning(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, lengths):
        pass

    @abstractmethod
    def reset_intermediate_vars(self):
        pass

    @abstractmethod
    def detach_intermediate_vars(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def loss(self, predictions, y, mask):
        predictions = predictions.view(-1, predictions.size(2))
        predictions *= torch.stack([mask] * predictions.size(1)).transpose(0, 1).float()
        return self.loss_func(predictions, y)

    def step_loss(self, input_x, target_y, lengths):
        predictions = self(input_x, lengths)

        # mask out padded values
        target_y = target_y.view(-1)  # Flatten out the batch
        mask = target_y != self.padding_idx
        target_y *= mask.long()

        return self.loss(predictions, target_y, mask)

    def training_step(self, train_batch, batch_idx):
        self.reset_intermediate_vars()
        x_i, y_i, l_i = train_batch
        loss = self.step_loss(x_i, y_i, l_i)
        self.detach_intermediate_vars()
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x_i, y_i, l_i = val_batch
        tmp_hidden = self.hidden
        # tmp_loss_func = self.loss_func
        self.reset_intermediate_vars()
        # self.loss_func = nn.CrossEntropyLoss(reduction="sum")
        cross_entropy = self.step_loss(x_i, y_i, l_i)
        self.detach_intermediate_vars()
        self.hidden = tmp_hidden
        perplexity = math.exp(cross_entropy.item())
        bpc = np.log2(perplexity)

        self.log(
            "loss/valid",
            cross_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "bpc/valid",
            bpc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # self.loss_func = tmp_loss_func

        return cross_entropy

    def test_step(self, batch, batch_idx):
        x_i, y_i, l_i = batch
        tmp_hidden = self.hidden
        # tmp_loss_func = self.loss_func
        self.reset_intermediate_vars()
        # self.loss_func = nn.CrossEntropyLoss(reduction="sum")

        cross_entropy = self.step_loss(x_i, y_i, l_i)
        self.detach_intermediate_vars()
        self.hidden = tmp_hidden
        perplexity = math.exp(cross_entropy.item())
        bpc = np.log2(perplexity)

        self.log(
            "loss/test",
            cross_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "bpc/test",
            bpc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
