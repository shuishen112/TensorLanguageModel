class RNN(nn.Module):

    # you can also accept arguments in your model constructor

    #  we don't use the output in this implemention
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output
    def initHidden(self,batch_size):
        return torch.zeros(batch_size,self.hidden_size).to(device)
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
        

class Model(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim,n_layers):
        super(Model,self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

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
        hidden = torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(device)
        return hidden