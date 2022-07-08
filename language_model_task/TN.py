import torch
import torch.nn as nn
from dataloader import Encoder
class TN(nn.Module):

    # tensor network unit
    def __init__(self, rank, output_size):
        super(TN, self).__init__()

        self.rank = rank
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.i2h = nn.Linear(self.rank, self.rank)
        self.h2o = nn.Linear(self.rank, output_size)
    

    def forward(self, data, m):
        # input = torch.cat((data, m.squeeze(1)), 1)

        # hidden = self.i2h(input)
        # output = self.h2o(hidden)

        # unit = self.i2h(data)
        unit = data.contiguous().view(-1,self.rank,self.rank)
        # get hidden
        activition = torch.nn.Tanh()
        # batch_size = unit.size(0)

        # weight = self.i2h.weight.unsqueeze(0).repeat([batch_size,1,1])
        # unit = torch.einsum("bij,bjk->bik",[unit,weight])
        m = activition(torch.einsum("bij,bjk->bik",[m,unit]))
        
        # # m = unit
        hidden = self.i2h(m)
        output = self.h2o(hidden)
        return hidden, output

    def init_m1(self):
        return nn.Linear(1,self.rank).to(self.device)
    def init_m2(self):
        return nn.Linear(self.rank, self.output_size)
    def init_hidden(self,batch_size):
        return torch.zeros(batch_size,self.rank).to(self.device)

class TN_layer(nn.Module):
    def __init__(self,rank,output_size):
        super(TN_layer,self).__init__()
        self.tn = TN(rank,output_size)
        self.rank = rank
        self.embedding = nn.Embedding(output_size,self.rank * self.rank)

        
    def forward(self,x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        encoding = self.embedding(x)
        
        # m = self.tn.init_hidden(batch_size)
        m = self.tn.init_m1()
        m = m.weight.view(-1,self.rank).unsqueeze(0).repeat([batch_size,1,1])
        hiddens = []
        # recurrent tn
        for i in range(seq_len):
            m, output = self.tn(encoding[:,i,:], m)
            hiddens.append(m)
        final_hidden = m
        hidden_tensor = torch.cat(hiddens,1)
        return hidden_tensor,final_hidden
        

class TN_model(nn.Module):
    def __init__(self,rank,output_size,):
        super(TN_model,self).__init__()

        self.rank = rank
        self.output_size = output_size
        
        self.tn = TN_layer(self.rank,output_size)
        self.fc = nn.Linear(self.rank,output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,x):
        out, hidden = self.tn(x)
        out = out.contiguous().view(-1,self.rank)
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
