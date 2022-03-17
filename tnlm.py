# %%
from turtle import TNavigator
import torch
from torch import nn

import numpy as np

# %%
text = ['hey how are you','good i am fine','have a nice day']

chars = set(''.join(text))
int2char = dict(enumerate(chars))
char2int = {char: ind for ind,char in int2char.items()}

# %%
maxlen = len(max(text, key=len))
print("The longest string has {} characters".format(maxlen))

# %%
# padding the text
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '

# %%
input_seq = []
target_seq = []

for i in range(len(text)):
    # remove the first token

    input_seq.append(text[i][:-1])
    target_seq.append(text[i][1:])
    print("Input Sequence:{}\n Target Sequence:{}".format(
        input_seq[i],
        target_seq[i]
    ))

# %%
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

# %%
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence,dict_size,seq_len,batch_size):
    features = np.zeros((batch_size,seq_len,dict_size),dtype = np.float32)

    for i in range(batch_size):
        for u in range(seq_len):
            features[i,u,sequence[i][u]] = 1
    return features


# %%
input_seq = torch.tensor(input_seq)
target_seq = torch.tensor(target_seq)

print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))


# %%
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# %%

class TN(nn.Module):

    # tensor network unit
    def __init__(self, rank, output_size):
        super(TN, self).__init__()

        self.rank = rank
        self.output_size = output_size
        input_size = rank + rank

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
        m = torch.einsum("bij,bjk->bik",[m,unit])
        # # m = unit
        hidden = activition(self.i2h(m))
        output = self.h2o(hidden)
        return hidden, output

    def init_m1(self):
        return nn.Linear(1,self.rank).to(device)
    def init_m2(self):
        return nn.Linear(self.rank, self.output_size)
    def init_hidden(self,batch_size):
        return torch.zeros(batch_size,self.rank).to(device)

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
        

class Model(nn.Module):
    def __init__(self,rank,output_size,):
        super(Model,self).__init__()

        self.rank = rank

        self.tn = TN_layer(self.rank,output_size)
        self.fc = nn.Linear(self.rank,output_size)

    def forward(self,x):
        out, hidden = self.tn(x)
        out = out.contiguous().view(-1,self.rank)
        out = self.fc(out)
        return out, hidden
    def init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(device)
        return hidden


# %%
model = Model(rank = 12,output_size = dict_size)

model = model.to(device)

n_epochs = 1000
lr = 0.01

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)




# %%
input_seq = input_seq.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq)
    output = output.to(device)
    target_seq = target_seq.to(device)
    loss = criterion(output.view(-1,dict_size), target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

# %%
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = torch.from_numpy(character)
    character = character.to(device) 
    out, hidden = model(character)
    out = out.squeeze(0)
    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()
    print(char_ind)

    return int2char[char_ind], hidden
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

# %%
print(sample(model, 15, 'hey'))

# %% [markdown]
# 

# %%



