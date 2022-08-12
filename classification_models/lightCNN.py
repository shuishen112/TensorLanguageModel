from classification_models.base_model import Baselighting, cal_accuracy
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, vocab_dim, e_dim, h_dim, o_dim):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(vocab_dim, e_dim, padding_idx=0)
        # self.emb.load_state_dict({"weight": torch.tensor(embedding)})
        non_trainable = True
        if non_trainable:
            self.emb.weight.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, h_dim, (3, e_dim))
        self.conv2 = nn.Conv2d(1, h_dim, (4, e_dim))
        self.conv3 = nn.Conv2d(1, h_dim, (5, e_dim))
        self.fc = nn.Linear(h_dim * 3, o_dim)

    def forward(self, x):
        embed = self.dropout(self.emb(x)).unsqueeze(1)
        c1 = torch.relu(self.conv1(embed).squeeze(3))
        p1 = torch.max_pool1d(c1, c1.size()[2]).squeeze(2)
        c2 = torch.relu(self.conv2(embed).squeeze(3))
        p2 = torch.max_pool1d(c2, c2.size()[2]).squeeze(2)
        c3 = torch.relu(self.conv3(embed).squeeze(3))
        p3 = torch.max_pool1d(c3, c3.size()[2]).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        hidden = self.fc(pool)

        return hidden


class litCNN(Baselighting):
    def __init__(self, vocab_dim, e_dim, h_dim, o_dim):
        super().__init__()
        self.model = CNN(vocab_dim, e_dim, h_dim, o_dim)
        # find the batch_size
        self.save_hyperparameters()

    def forward(self, x):
        encode = self.model(x)
        return encode

    # train and validation
    def training_step(self, train_batch, batch_idx):
        text, label, lengths = train_batch
        predictions = self.model(text)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log("train_loss", loss)
        self.log("acc", acc)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, val_batch, batch_idx):
        text, label, lengths = val_batch
        predictions = self.model(text)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return acc
