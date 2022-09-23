from torch import nn
import torch
import math
from torch.nn import Parameter
from torch import Tensor


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


class MIRNNCell(nn.Module):
    """Multiplicative integration cell"""

    def __init__(self, embed_size, hidden_size, output_size):
        super(MIRNNCell, self).__init__()
        input_size = embed_size + hidden_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, embed_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.bias_hh = Parameter(torch.randn(hidden_size))

        self.h2o = nn.Linear(input_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input: Tensor, state: Tensor):

        input_ = torch.cat((input, state), 1)
        hx = state
        hidden = (torch.mm(input, self.weight_ih.t()) + self.bias_ih) * (
            torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        )

        hy = torch.tanh(hidden)

        output = self.h2o(input_)
        return output, hy

    def initHidden(self, batch_size):
        # return torch.zeros(batch_size,self.hidden_size).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
            self.device
        )


class MRNNCell(nn.Module):
    #  implement of multiplicative RNN
    def __init__(self, embed_size, hidden_size, output_size):
        super(MRNNCell, self).__init__()

        self.hidden_size = hidden_size
        input_size = embed_size + hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.i2h = nn.Linear(input_size, hidden_size)
        self.Wih = nn.Linear(embed_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size, output_size)

        self.w_im = nn.Linear(embed_size, hidden_size)
        self.w_hm = nn.Linear(hidden_size, hidden_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)

        mx = self.w_im(data) * self.w_hm(last_hidden)

        wi = self.Wih(data)
        wh = self.Whh(mx)

        hidden = torch.tanh(wi + wh)
        output = self.h2o(input)
        return output, hidden

    def initHidden(self, batch_size):
        # return torch.zeros(batch_size,self.hidden_size).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
            self.device
        )


class SecondOrderCell(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size):
        super(SecondOrderCell, self).__init__()
        # tensor network unit
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.three_order = nn.Bilinear(embed_size, self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, output_size)

    def forward(self, data, last_hidden):

        hidden = nn.Tanh(self.three_order(data, last_hidden))
        output = self.h2o(hidden)

        return output,hidden

    def initHidden(self, batch_size):
        # return torch.ones(batch_size, 1, self.rank).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
            self.device
        )


class RACs(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size):
        super(RACs, self).__init__()

        self.hidden_size = hidden_size
        input_size = embed_size + hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Wih = nn.Linear(embed_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        wi = self.Wih(data)
        wh = self.Whh(last_hidden)
        hidden = wi * wh
        output = self.h2o(input)
        return output, hidden

    def initHidden(self, batch_size):
        # return torch.zeros(batch_size,self.hidden_size).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
            self.device
        )
