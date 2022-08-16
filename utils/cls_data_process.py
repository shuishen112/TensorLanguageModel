from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from torch.utils.data import DataLoader
import torch
from config import args
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("glue", "sst2")
tokenizer = get_tokenizer("basic_english")
max_length = 20
batch_size = args.batch_size


def get_alphabet(corpuses):
    """
    obtain the dict
                    :param corpuses:
    """
    word_counter = Counter()

    for corpus in corpuses:
        for item in corpus:
            tokens = tokenizer(item["sentence"])
            for token in tokens:
                word_counter[token] += 1
    print("there are {} words in dict".format(len(word_counter)))
    # logging.info("there are {} words in dict".format(len(word_counter)))
    word_dict = {word: e + 2 for e, word in enumerate(list(word_counter))}
    word_dict["UNK"] = 1
    word_dict["<PAD>"] = 0

    return word_dict


vocab = get_alphabet([dataset["train"], dataset["validation"]])


def get_embedding(alphabet, filename="", embedding_size=100):
    embedding = np.random.rand(len(alphabet), embedding_size)
    if filename is None:
        return embedding
    with open(filename, encoding="utf-8") as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print("epch %d" % i)
            items = line.strip().split(" ")
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]
                print((vocab_size, embedding_size))
            else:
                word = items[0]
                if word in alphabet:
                    embedding[alphabet[word]] = items[1:]

    print("done")
    return embedding


embedding = get_embedding(
    vocab, filename="embedding/glove.6B.300d.txt", embedding_size=300
)


def convert_to_word_ids(sentence, alphabet, max_len=40):
    """
    docstring here
            :param sentence:
            :param alphabet:
            :param max_len=40:
    """
    indices = []
    tokens = tokenizer(sentence)

    for word in tokens:
        if word in alphabet:
            indices.append(alphabet[word])
        else:
            continue
    result = indices + [alphabet["<PAD>"]] * (max_len - len(indices))

    return result[:max_len], min(len(indices), max_len)


class DataMaper(Dataset):
    def __init__(self, dataset, vocab, max_length=20):
        self.x = dataset["sentence"]
        self.y = dataset["label"]
        self.max_length = max_length
        self.vocab = vocab

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sentence = self.x[idx]
        label = self.y[idx]

        enc_sentence, lengths = convert_to_word_ids(
            sentence, self.vocab, max_len=self.max_length
        )
        t_sentence = torch.tensor(enc_sentence).to(device)
        t_label = torch.tensor(label).to(device)
        t_length = torch.tensor(lengths).to(device)
        return t_sentence, t_label, t_length


# train = DataMaper(dataset["train"], vocab, max_length)
train_df = pd.read_csv("data/glue_data/SST-2/train_aug.tsv", "\t")
train = DataMaper(train_df, vocab, max_length)
# train = DataMaper(dataset["train"], vocab, max_length)

validation = DataMaper(dataset["validation"], vocab, max_length)
test = DataMaper(dataset["test"], vocab, max_length)

loader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
loader_validation = DataLoader(validation, batch_size=batch_size)
loader_test = DataLoader(test, batch_size=batch_size)
