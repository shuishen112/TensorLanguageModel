import torch.nn as nn
import torch

from dataloader import Encoder

class RNN(nn.Module):

    # you can also accept arguments in your model constructor

    #  we don't use the output in this implemention
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output
    def initHidden(self,batch_size):
        return torch.zeros(batch_size,self.hidden_size).to(self.device)
class RNN_layer(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim):
        super(RNN_layer,self).__init__()
        self.rnn = RNN(input_size,hidden_dim,output_size)
        self.embedding = nn.Embedding(input_size, input_size)
        
    def forward(self,x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = self.embedding(x)

        hidden = self.rnn.initHidden(batch_size)
        hiddens = []
        # recurrent rnn
        for i in range(seq_len):
            hidden, output = self.rnn(x[:,i,:], hidden)
            hiddens.append(hidden.unsqueeze(1))
        final_hidden = hidden
        hidden_tensor = torch.cat(hiddens,1)
        return hidden_tensor,final_hidden
        

class RNN_Model(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim,n_layers):
        super(RNN_Model,self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.output_size = output_size
        # define the layer
        # self.rnn = nn.RNN(input_size,hidden_dim,n_layers,batch_first= True)
        self.rnn = RNN_layer(input_size,output_size,hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_size)

    def forward(self,x):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        # passing in the input and hidden state into the rnn model
        # out,hidden = self.rnn(x,hidden)
        out, hidden = self.rnn(x)
        out = out.contiguous().view(-1,self.hidden_dim)
        out = self.fc(out)

        return out, hidden
    def init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(self.device)
        return hidden
    def loss(self):
        return nn.CrossEntropyLoss()
    
    def predict(self, encoder: Encoder, character):
        # One-hot encoding our input to fit into the model
        starting_symbols = encoder.map_tokens_to_ids(character)
        starting_symbols_tensor = torch.tensor(starting_symbols).to(self.device)
        starting_symbols_tensor = starting_symbols_tensor.view(-1,starting_symbols_tensor.size(0))
        
        
        out, hidden = self(starting_symbols_tensor)

        prob = nn.functional.softmax(out[-1], dim=0).data
        # Taking the class with the highest probability score from the output
        char_ind = torch.max(prob, dim=0)[1].item()

        return encoder.id_to_token(char_ind), hidden
        
    def generate_text(self, encoder: Encoder, start, out_len):
        start = start.lower()
        # First off, run through the starting characters
        chars = [ch for ch in start]
        size = out_len - len(chars)
        for ii in range(size):
            char, h = self.predict(encoder, chars)
            chars.append(char)
        return ''.join(chars)