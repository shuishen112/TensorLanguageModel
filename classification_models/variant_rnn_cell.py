from torch import nn
import torch


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


class mRNN(nn.Module):
    #  implement of multiplicative RNN
    def __init__(self, embed_size, hidden_size, output_size):
        super(mRNN, self).__init__()

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


class TensorRAC(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size):
        super(TensorRAC, self).__init__()

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
        activation = torch.nn.ReLU()
        hidden = activation(wi * wh)
        output = self.h2o(input)
        return output, hidden

    def initHidden(self, batch_size):
        # return torch.zeros(batch_size,self.hidden_size).to(self.device)
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
            self.device
        )
