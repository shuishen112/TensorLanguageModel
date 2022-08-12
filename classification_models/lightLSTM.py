from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch
from torch import nn
from base_model import Baselighthing


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

        # self.embedding.load_state_dict({"weight": torch.tensor(embedding)})
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        return h.to(self.device), c.to(self.device)

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


class litRNN(Baselighthing):
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
