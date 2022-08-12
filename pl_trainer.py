# %% [markdown]
# ## Using pytorch lightning to train the model

# %% [markdown]
# ## get dataset

# %%
from datasets import load_dataset

import torch

# preprocessing and tokenizer
from collections import Counter
from torchtext.data.utils import get_tokenizer
import wandb
from TN_module import TN_model_for_classfication

dataset = load_dataset("glue", "sst2")
tokenizer = get_tokenizer("basic_english")


def get_alphabet(corpuses):
    """
    obtain the dict
                    :param corpuses:
    """
    word_counter = Counter()

    for corpus in corpuses:
        for item in corpus:
            tokens = tokenizer(item["sentence"])
            for token in tokens:
                word_counter[token] += 1
    print("there are {} words in dict".format(len(word_counter)))
    # logging.info("there are {} words in dict".format(len(word_counter)))
    word_dict = {word: e + 2 for e, word in enumerate(list(word_counter))}
    word_dict["UNK"] = 1
    word_dict["<PAD>"] = 0

    return word_dict


vocab = get_alphabet([dataset["train"], dataset["validation"]])


# %%
# get embedding
import numpy as np


def get_embedding(alphabet, filename="", embedding_size=100):
    embedding = np.random.rand(len(alphabet), embedding_size)
    if filename is None:
        return embedding
    with open(filename, encoding="utf-8") as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print("epch %d" % i)
            items = line.strip().split(" ")
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]
                print((vocab_size, embedding_size))
            else:
                word = items[0]
                if word in alphabet:
                    embedding[alphabet[word]] = items[1:]

    print("done")
    return embedding


embedding = get_embedding(
    vocab, filename="embedding/glove.6B.300d.txt", embedding_size=300
)

# %%
# convert to index


def convert_to_word_ids(sentence, alphabet, max_len=40):
    """
    docstring here
            :param sentence:
            :param alphabet:
            :param max_len=40:
    """
    indices = []
    tokens = tokenizer(sentence)

    for word in tokens:
        if word in alphabet:
            indices.append(alphabet[word])
        else:
            continue
    result = indices + [alphabet["<PAD>"]] * (max_len - len(indices))

    return result[:max_len], min(len(indices), max_len)


test_enc, length = convert_to_word_ids("hello, how are you", vocab, 10)
print(test_enc)
print(length)

# %%
# generate data batch and iterator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64


class DataMaper(Dataset):
    def __init__(self, dataset, vocab, max_length=20):
        self.x = dataset["sentence"]
        self.y = dataset["label"]
        self.max_length = max_length
        self.vocab = vocab

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sentence = self.x[idx]
        label = self.y[idx]

        enc_sentence, lengths = convert_to_word_ids(
            sentence, self.vocab, max_len=self.max_length
        )
        t_sentence = torch.tensor(enc_sentence).to(device)
        t_label = torch.tensor(label).to(device)
        t_length = torch.tensor(lengths).to(device)
        return t_sentence, t_label, t_length


max_length = 20
train = DataMaper(dataset["train"], vocab, max_length)
validation = DataMaper(dataset["validation"], vocab, max_length)
test = DataMaper(dataset["test"], vocab, max_length)

loader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
loader_validation = DataLoader(validation, batch_size=batch_size)
loader_test = DataLoader(test, batch_size=batch_size)

# %%
import torch
from torch import nn
import pytorch_lightning as pl


def cal_accuracy(probs, target):
    predictions = probs.argmax(dim=1)
    corrects = predictions == target
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy


# %%
# change the model to pytorch_lightning
class Baselignthing(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, lengths):
        encode = self.model(x, lengths)
        return encode

    # optimizers go into configure_optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # train and validation
    def training_step(self, train_batch, batch_idx):
        text, label, lengths = train_batch
        predictions = self.model(text, lengths)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log("train_loss", loss)
        self.log("acc", acc)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, val_batch, batch_idx):
        text, label, lengths = val_batch
        predictions = self.model(text, lengths)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(predictions, label)
        acc = cal_accuracy(predictions, label)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return acc

    def training_epoch_end(self, train_step_outputs) -> None:
        all_acc = torch.stack([x["train_acc"] for x in train_step_outputs])
        print("train_epoch_acc:", torch.mean(all_acc))
        self.log("train_epoch_acc", torch.mean(all_acc))

    def validation_epoch_end(self, validation_step_outputs):
        all_acc = torch.stack(validation_step_outputs)
        print("val_epoch_acc:", torch.mean(all_acc))
        self.log("val_epoch_acc", torch.mean(all_acc))


class CNN(nn.Module):
    def __init__(self, vocab_dim, e_dim, h_dim, o_dim):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(vocab_dim, e_dim, padding_idx=0)
        self.emb.load_state_dict({"weight": torch.tensor(embedding)})
        non_trainable = True
        if non_trainable:
            self.emb.weight.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, h_dim, (3, e_dim))
        self.conv2 = nn.Conv2d(1, h_dim, (4, e_dim))
        self.conv3 = nn.Conv2d(1, h_dim, (5, e_dim))
        self.fc = nn.Linear(h_dim * 3, o_dim)
        # self.softmax = nn.Softmax(dim=1)
        # self.log_softmax = nn.LogSoftmax(dim=1)

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
        # return self.softmax(hidden), self.log_softmax(hidden)
        return hidden


class litCNN(Baselignthing):
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


# %%

# %%
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class LSTM(nn.Module):

    # define all the layers used in model
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        lstm_layers,
        bidirectional,
        dropout,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.embedding.load_state_dict({"weight": torch.tensor(embedding)})
        non_trainable = True
        if non_trainable:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_dim * num_directions, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = num_directions
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size):
        h, c = (
            Variable(
                torch.zeros(
                    self.lstm_layers * self.num_directions, batch_size, self.hidden_dim
                )
            ),
            Variable(
                torch.zeros(
                    self.lstm_layers * self.num_directions, batch_size, self.hidden_dim
                )
            ),
        )
        return h.to(device), c.to(device)

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(
            embedded, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False
        )
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
        # output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        # if it is bi directional LSTM, we should concat the two f
        out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # out = h_n[-1]
        # print(h_n.shape)
        # out = output_unpacked[:, -1, :]
        preds = self.fc1(out)
        return preds


class litRNN(Baselignthing):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        lstm_layers,
        bidirectional,
        dropout,
        pad_index,
    ):
        super().__init__()
        self.model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_index=pad_index,
        )

        # find the batch_size
        self.save_hyperparameters()


class TensorRAC(nn.Module):
    # you can also accept arguments in your model constructor

    #  we don't use the output in this implemention
    def __init__(self, embed_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = embed_size + hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.i2h = nn.Linear(input_size, hidden_size)

        self.Wih = nn.Linear(embed_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

        self.h2o = nn.Linear(input_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        # hidden = torch.sigmoid(self.i2h(input))
        wi = torch.relu(self.Wih(data))
        wh = torch.relu(self.Whh(last_hidden))
        # hidden = torch.tanh(wi + wh)
        eps = 1e-7
        hidden = torch.tanh(torch.log(wi + eps) + torch.log(wh + eps))
        output = self.h2o(input)
        return output, hidden

    def initHidden(self, batch_size):
        # return torch.zeros(batch_size,self.hidden_size).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
            self.device
        )


class RNN(nn.Module):

    # you can also accept arguments in your model constructor

    #  we don't use the output in this implemention
    def __init__(self, embed_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = embed_size + hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.i2h = nn.Linear(input_size, hidden_size)

        self.Wih = nn.Linear(embed_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

        self.h2o = nn.Linear(input_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        # hidden = torch.sigmoid(self.i2h(input))
        wi = self.Wih(data)
        wh = self.Whh(last_hidden)
        hidden = torch.tanh(wi + wh)
        output = self.h2o(input)
        return output, hidden

    def initHidden(self, batch_size):
        # return torch.zeros(batch_size,self.hidden_size).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
            self.device
        )


class RNN_layer(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, output_size):
        super(RNN_layer, self).__init__()
        self.rnn = RNN(embed_size, hidden_dim, output_size)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, text_lens):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = self.dropout(self.embedding(x))

        hidden = self.rnn.initHidden(batch_size)
        hiddens = []
        # recurrent rnn
        for i in range(seq_len):
            output, hidden_next = self.rnn(x[:, i, :], hidden)
            mask = (
                (i < text_lens).float().unsqueeze(1).expand_as(hidden_next).to(device)
            )
            hidden_next = (hidden_next * mask + hidden * (1 - mask)).to(device)
            hiddens.append(hidden_next.unsqueeze(1))
            hidden = hidden_next
        final_hidden = hidden
        hidden_tensor = torch.cat(hiddens, 1)
        return hidden_tensor, final_hidden, output


class RNN_Model_for_classfication(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, output_size):
        super(RNN_Model_for_classfication, self).__init__()

        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = vocab_size
        # define the layer
        # self.rnn = nn.RNN(embed_size,hidden_dim,num_layers = 1,batch_first= True)
        self.rnn = RNN_layer(self.vocab_size, embed_size, hidden_dim, output_size)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, lens):

        hidden_tensor, final_hidden, output = self.rnn(x, lens)

        out = self.fc(final_hidden)
        return out


class litSimpleRNN(Baselignthing):
    def __init__(self, vocab_size, embed_size, hidden_dim, output_size):
        super().__init__()
        self.model = RNN_Model_for_classfication(
            vocab_size, embed_size, hidden_dim, output_size
        )

        # find the batch_size
        self.save_hyperparameters()


class litTNLM(Baselignthing):
    def __init__(self, rank, vocab_size, output_size):
        super().__init__()
        self.model = TN_model_for_classfication(
            rank=rank, vocab_size=vocab_size, output_size=output_size
        )


# %%
# wandb_logger = WandbLogger(name="TNLMAAAI", project="text_classification")


# model = litCNN(len(vocab),e_dim = 300,h_dim = 64, o_dim = 2)

# model = litRNN(
#     vocab_size=len(vocab),
#     embedding_dim=300,
#     hidden_dim=100,
#     num_classes=2,
#     lstm_layers=2,
#     bidirectional=True,
#     dropout=0.5,
#     pad_index=0,
# )

# model = litSimpleRNN(
#     vocab_size=len(vocab), embed_size=300, hidden_dim=256, output_size=2
# )
model = litTNLM(rank=5, vocab_size=len(vocab), output_size=2)
# wandb_logger.watch(model, log="all")
trainer = pl.Trainer(logger=None, max_epochs=20, accelerator="gpu")
trainer.fit(model, loader_train, loader_validation)
# wandb.finish()

# %%
