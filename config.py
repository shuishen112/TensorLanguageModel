import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
    "--embed_size",
    default=300,
    type=int,
    help="embedding size",
)
parser.add_argument(
    "--batch_size",
    default=64,
    type=int,
    help="batch_size",
)
parser.add_argument(
    "--epoch",
    default=30,
    type=int,
    help="epoch",
)
parser.add_argument(
    "--hidden_size",
    default=64,
    type=int,
    help="hidden_size",
)

parser.add_argument(
    "--output_size",
    default=2,
    type=int,
    help="hidden_size",
)
parser.add_argument(
    "--rank",
    default=3,
    type=int,
    help="rank of tensor train",
)

parser.add_argument(
    "--activation",
    default="nn.GELU",
    help="the activation in TNLM[nn.LeakyReLU,nn.RReLU,nn.ReLU,nn.ReLU6,nn.SELU,nn.GELU]",
)

parser.add_argument(
    "--max_length",
    default=20,
    type=int,
    help="max length of sentence",
)

parser.add_argument(
    "--dropout",
    default=0.2,
    type=float,
    help="dropout",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for initialization",
)
parser.add_argument(
    "--data_name",
    default="mpqa",
    help=" dataset name",
)

parser.add_argument(
    "--cell",
    default="LSTM",
    help=" cell name: CNN, Second, TinyTNLM, RNN, MRNN, MIRNN",
)
args = parser.parse_args()
print(args)
