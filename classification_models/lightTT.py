from classification_models.base_model import Baselighting, cal_accuracy
import torch
import torch.nn as nn


active = {
    "nn.ReLU": nn.ReLU(),
    "nn.RReLU": nn.RReLU(),
    "nn.LeakyReLU": nn.LeakyReLU(),
    "nn.ReLU6": nn.ReLU6(),
    "nn.SELU": nn.SELU(),
    "nn.GELU": nn.GELU(),
}


class TN_3order(nn.Module):
    def __init__(self, rank, output_size, activation):
        super(TN_3order, self).__init__()
        # tensor network unit
        self.output_size = output_size
        self.rank = rank
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.three_order = nn.init.orthogonal_(
        #     nn.Parameter(torch.zeros(self.rank, self.rank, self.rank * self.rank))
        # ).to(self.device)
        self.three_order = nn.Bilinear(self.rank, self.rank * self.rank, self.rank)
        self.h2o = nn.Linear(self.rank, output_size)

    def forward(self, data, m):

        unit = data
        h = m.squeeze(1)
        activation = active[self.activation]
        hidden = activation(self.three_order(h, unit))
        # hidden = activation(nn.functional.bilinear(h, unit, self.three_order))
        output = self.h2o(hidden)
        hidden = hidden.unsqueeze(1)
        return hidden, output

    def init_m1(self, batch_size):
        # return torch.ones(batch_size, 1, self.rank).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, 1, self.rank)).to(
            self.device
        )
        # return nn.Linear(1,self.rank).to(self.device)


class TN(nn.Module):

    # tensor network unit
    def __init__(self, rank, output_size, activation):
        super(TN, self).__init__()

        self.rank = rank
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        # self.i2h = nn.Linear(self.rank,self.rank)
        self.wih = nn.Linear(self.rank, self.rank)
        self.whh = nn.Linear(self.rank * self.rank, self.rank * self.rank)
        self.h2o = nn.Linear(self.rank, output_size)
        self.activation = activation

    def forward(self, data, m):

        # unit = data.contiguous().view(-1, self.rank, self.rank)
        # unit = data.contiguous.view(-1, self.rank)
        batch_size = data.size()[0]
        unit = data
        # get hidden

        activition = active[self.activation]
        # batch_size = unit.size(0)

        w1 = self.wih(m.squeeze(1))
        w2 = self.whh(unit).view(batch_size, self.rank, self.rank)
        # w2 = unit.view(batch_size, self.rank, self.rank)
        hidden = activition(
            torch.einsum("bij,bjk->bik", [w1.unsqueeze(1), w2])
        )  # [batch, 1, rank] []

        # # m = unit
        # hidden = self.i2h(m)
        output = self.h2o(hidden)
        return hidden, output

    def init_m1(self, batch_size):
        # return torch.ones(batch_size, 1, self.rank).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, 1, self.rank)).to(
            self.device
        )
        # return nn.Linear(1,self.rank).to(self.device)

    def init_m2(self):
        return nn.Linear(self.rank, self.output_size)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.rank).to(self.device)


class TN_layer(nn.Module):
    def __init__(
        self, rank, vocab_size, output_size, dropout=0.2, activation="nn.RELU"
    ):
        super(TN_layer, self).__init__()
        self.tn = TN(rank, output_size, activation)
        self.rank = rank
        self.embedding = nn.Embedding(vocab_size, self.rank * self.rank, padding_idx=0)

        # self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_lens):
        batch_size = x.size(0)
        seq_len = x.size(1)

        encoding = self.embedding(x)

        # m = self.tn.init_hidden(batch_size)
        m = self.tn.init_m1(batch_size)
        # m = m.weight.view(-1,self.rank).unsqueeze(0).repeat([batch_size,1,1])
        hiddens = []
        # recurrent tn
        for i in range(seq_len):
            hidden_next, output = self.tn(encoding[:, i, :], m)
            mask = (
                (i < text_lens).float().unsqueeze(1).unsqueeze(1).expand_as(hidden_next)
            )
            hidden_next = hidden_next * mask + m * (1 - mask)
            hiddens.append(hidden_next.unsqueeze(1))
            m = hidden_next

            # hiddens.append(m)
        final_hidden = m
        hidden_tensor = torch.cat(hiddens, 1)
        return hidden_tensor, final_hidden


class TN_model_for_classfication(nn.Module):
    def __init__(self, rank, vocab_size, output_size, dropout, activation):
        super(TN_model_for_classfication, self).__init__()

        self.rank = rank
        self.output_size = output_size
        self.vocab_size = vocab_size

        self.tn = TN_layer(self.rank, self.vocab_size, output_size, dropout, activation)
        self.fc = nn.Linear(self.rank, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, lens):
        seq_output, hidden = self.tn(x, lens)
        # out = out.contiguous().view(-1,self.rank)
        output = self.fc(hidden.squeeze(1))

        return output

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden


class litTNLM(Baselighting):
    def __init__(self, rank, vocab_size, output_size, dropout, activation):
        super().__init__()
        self.model = TN_model_for_classfication(
            rank=rank,
            vocab_size=vocab_size,
            output_size=output_size,
            dropout=dropout,
            activation=activation,
        )
