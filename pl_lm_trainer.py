import math
import time

import torch
import torch.nn as nn

from language_models.RNN import RNN_language_model
from language_models.simpleRNN import RNNModel
from lm_config import args
from utils.lm_data_process import corpus, get_batch, test_data, train_data, val_data

ntokens = len(corpus.dictionary)

# model = RNNModel(
#     "LSTM",
#     ntokens,
#     args.embed_size,
#     args.hidden_size,
#     args.nlayers,
#     args.dropout,
#     args.tied,
# ).to(device)

model = RNN_language_model(ntokens, args.embed_size, args.hidden_size, 0).to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())

clip = 0.25
lr = 20
log_interval = 200
epochs = 10
args.save = "model.pt"
best_val_loss = None


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary)
    # hidden = model.init_hidden(args.eval_batch_size)
    hidden = model.rnn_cell.initHidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.batch_size):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.

    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    # hidden = model.init_hidden(args.batch_size)
    hidden = model.rnn_cell.initHidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.batch_size)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        # model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # for name, param in model.named_parameters():
        #     print(name, param.grad)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.batch_size,
                    lr,
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(val_data)
    print("-" * 89)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
        "valid ppl {:8.2f}".format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
        )
    )
    print("--" * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, "wb") as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0


# Load the best saved model.
with open(args.save, "rb") as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    # if args.model in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
    #     model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print("=" * 89)
print(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)
    )
)
print("=" * 89)

# if len(args.onnx_export) > 0:
#     # Export the model in ONNX format.
#     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
