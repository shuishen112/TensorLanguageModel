from torch import nn
import torch


class TensorRAC(nn.Module):
    # you can also accept arguments in your model constructor

    #  we don't use the output in this implemention
    def __init__(self, embed_size, hidden_size, output_size):
        super(TensorRAC, self).__init__()

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
