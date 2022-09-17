from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer

from config import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer("basic_english")


def get_alphabet(corpuses):
    """
    obtain the dict
                    :param corpuses:
    """
    word_counter = Counter()

    for corpus in corpuses:
        for _, item in corpus.iterrows():
            tokens = tokenizer(item["sentence"])
            for token in tokens:
                word_counter[token] += 1
    print("there are {} words in dict".format(len(word_counter)))
    # logging.info("there are {} words in dict".format(len(word_counter)))
    word_dict = {word: e + 2 for e, word in enumerate(list(word_counter))}
    word_dict["UNK"] = 1
    word_dict["<PAD>"] = 0

    return word_dict


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


# embedding = get_embedding(
#     vocab, filename="embedding/glove.6B.300d.txt", embedding_size=300
# )


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
        self.x = dataset["sentence"].to_list()
        self.y = dataset["label"].to_list()
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


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_name: str = "sst2"):
        super().__init__()
        df_train = pd.read_csv(
            "data/classfication_data/sst2/train.csv",
            sep="\t",
            names=["sentence", "label"],
        )
        df_validation = pd.read_csv(
            "data/classfication_data/sst2/dev.csv",
            sep="\t",
            names=["sentence", "label"],
        )
        df_test = pd.read_csv(
            "data/classfication_data/sst2/test.csv",
            sep="\t",
            names=["sentence", "label"],
        )
        self.vocab = get_alphabet([df_train, df_validation])

        self.train = DataMaper(df_train, self.vocab, args.max_length)
        self.validation = DataMaper(df_validation, self.vocab, args.max_length)
        self.test = DataMaper(df_test, self.vocab, args.max_length)
        self.data_name = data_name

    def setup(self, stage: Optional[str] = None) -> None:

        # load dataset
        pass

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=args.batch_size)
