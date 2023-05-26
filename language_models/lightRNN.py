import math
from typing import List

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from language_models.language_base_model import LanguageBaselightning



class RNNCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize the weights with random numbers.
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(hidden_size)) # input to hidden
        self.bias_hh = Parameter(torch.randn(hidden_size)) # hidden to hidden
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        # input is the input at the current timestep
        # state is the hidden state from the previous timestep
        hx = state
        hidden = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        hy = torch.tanh(hidden)
        return hy


class RNNLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        inputs = input.unbind(1)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state]
        return torch.stack(outputs, 1), state


class JitRNN_language_model(LanguageBaselightning):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        padding_idx: int,
        learning_rate: int = 0.001,
    ):
        super(JitRNN_language_model, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.padding_idx = torch.tensor(padding_idx).to(self.device)
        self.padding_idx = torch.tensor(padding_idx)
        self.learning_rate = learning_rate

        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=self.padding_idx
        )

        self.dense = nn.Linear(hidden_size, embedding_size)

        self.rnn = RNNLayer(RNNCell, embedding_size, hidden_size)

        self.output_layer = nn.Linear(embedding_size, vocab_size)
        self.hidden = None

        # tie the weights of the output embeddings with the input embeddings
        self.output_layer.weight = self.embedding.weight
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, seq_length):
        batch_size, seq_length = x.size()
        # get embedding encoder
        x = self.embedding(x)
        # get output of rnn
        self.hidden = torch.zeros(batch_size, self.hidden_size).type_as(x)

        output, self.hidden = self.rnn(x, self.hidden)

        out = self.dense(output)
        out = self.output_layer(out)
        return out.view(
            batch_size, seq_length, self.vocab_size
        )  # Dimensions -> Batch x Sequence x Vocab

    def reset_intermediate_vars(self):
        self.hidden = None

    def detach_intermediate_vars(self):
        self.hidden = self.hidden.detach()


# class RNN(nn.Module):

#     # you can also accept arguments in your model constructor

#     #  we don't use the output in this implemention
#     def __init__(
#         self,
#         embed_size,
#         hidden_size,
#     ):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size
#         # input_size = embed_size + hidden_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.i2h = nn.Linear(input_size, hidden_size)
#         self.Wih = nn.Linear(embed_size, hidden_size)
#         self.Whh = nn.Linear(hidden_size, hidden_size)
#         # self.h2o = nn.Linear(input_size, output_size)

#     def forward(self, data, last_hidden):

#         wi = self.Wih(data)
#         wh = self.Whh(last_hidden)
#         hidden = torch.relu(wi + wh)
#         # output = self.h2o(input)
#         return hidden

#     def initHidden(self, batch_size):
#         # return torch.zeros(batch_size,self.hidden_size).to(self.device)
#         return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)).to(
#             self.device
#         )

# class RNN_language_model(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,
#         embed_size: int,
#         hidden_size: int,
#         padding_idx: int,
#     ):
#         super(RNN_language_model, self).__init__()
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.padding_idx = torch.tensor(padding_idx).to(self.device)

#         self.embedding = nn.Embedding(
#             vocab_size, embed_size, padding_idx=self.padding_idx
#         )

#         self.dense = nn.Linear(hidden_size, embed_size)
#         # note that output_size = vocab_size
#         self.rnn_cell = RNN(
#             embed_size,
#             hidden_size,
#         )

#         self.output_layer = nn.Linear(embed_size, vocab_size)

#         # tie the weights of the output embeddings with the input embeddings
#         # self.output_layer.weight = self.embedding.weight
#         self.loss_func = nn.CrossEntropyLoss()

#     def forward(self, x, seq_length):
#         batch_size, seq_length = x.size()
#         # get embedding encoder
#         x = self.embedding(x)
#         # get output of rnn
#         self.hidden = self.rnn_cell.initHidden(batch_size)

#         hiddens = []
#         # recurrent rnn
#         for i in range(seq_length):
#             hidden_next = self.rnn_cell(x[:, i, :], self.hidden)
#             hiddens.append(hidden_next.unsqueeze(1))
#             self.hidden = hidden_next
#         hidden_tensor = torch.cat(hiddens, 1)
#         out = hidden_tensor.contiguous().view(-1, self.hidden_size)
#         out = self.dense(out)
#         out = self.output_layer(out)
#         return (
#             out.view(batch_size, seq_length, self.vocab_size),
#             self.hidden,
#         )  # Dimensions -> Batch x Sequence x Vocab

#     def loss(self, predictions, y, mask):
#         predictions = predictions.view(-1, predictions.size(2))
#         predictions *= torch.stack([mask] * predictions.size(1)).transpose(0, 1).float()
#         return self.loss_func(predictions, y)
