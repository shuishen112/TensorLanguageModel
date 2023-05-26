from classification_models.base_model import Baselighting, cal_accuracy
import torch
from torch import nn
class litWord2vec(Baselighting):
    def __init__(self, vocab_dim, e_dim, o_dim, embedding=None):
        super().__init__()
        # find the batch_size
        self.emb = nn.Embedding(vocab_dim, e_dim, padding_idx=0)
        self.emb.load_state_dict({"weight": torch.tensor(embedding)})

        non_trainable = True    
        if non_trainable:
            self.emb.weight.requires_grad = False
        self.fc = nn.Linear(e_dim, o_dim)
        self.save_hyperparameters()

    def forward(self, x):
        encode = self.emb(x)
        # mean pooling
        encode = torch.mean(encode, dim=1)
        encode = self.fc(encode)

        return encode
    # train and validation
    def training_step(self, train_batch, batch_idx):
        text, label, lengths = train_batch
        predictions = self(text)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, val_batch, batch_idx):
        text, label, lengths = val_batch
        predictions = self(text)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return acc
    def test_step(self, batch, batch_idx):
        text, label, lengths = batch
        predictions = self(text)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return acc