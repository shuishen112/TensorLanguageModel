from torch import nn
import torch
from classification_models.base_model import Baselighting


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
                (i < text_lens)
                .float()
                .unsqueeze(1)
                .expand_as(hidden_next)
                .to(self.device)
            )
            hidden_next = (hidden_next * mask + hidden * (1 - mask)).to(self.device)
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


class litSimpleRNN(Baselighting):
    def __init__(self, vocab_size, embed_size, hidden_dim, output_size):
        super().__init__()
        self.model = RNN_Model_for_classfication(
            vocab_size, embed_size, hidden_dim, output_size
        )
        # find the batch_size
        self.save_hyperparameters()
