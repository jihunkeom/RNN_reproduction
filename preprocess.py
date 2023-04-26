import re
import unicodedata
import numpy
import torch
import pickle
import string
from collections import OrderedDict
from tqdm.auto import tqdm
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader

def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"

    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
        and c in all_letters
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z ]+", r" ", s)
    
    return s

def read_data(src, tgt):
    src_corpus, tgt_corpus = [], []

    with open("./data/en_de/train."+str(src)+".txt", 'r') as f:
        for line in f:
            text = normalizeString(line)
            src_corpus.append(text)

    with open("./data/en_de/train."+str(tgt)+".txt", 'r') as f:
        for line in f:
            text = normalizeString(line)
            tgt_corpus.append(text)

    
    pairs = []
    for i in range(len(src_corpus)):
        if (len(src_corpus[i].split()) > 0) and (len(src_corpus[i].split()) < 50):
            if (len(tgt_corpus[i].split()) > 0) and (len(tgt_corpus[i].split()) < 50):
                pairs.append([src_corpus[i], tgt_corpus[i]])

    return pairs

def read_test_data(src, tgt):
    src_corpus, tgt_corpus = [], []

    with open("./data/en_de/newstest2014."+str(src)+".txt", 'r') as f:
        for line in f:
            text = normalizeString(line)
            src_corpus.append(text)

    with open("./data/en_de/newstest2014."+str(tgt)+".txt", 'r') as f:
        for line in f:
            text = normalizeString(line)
            tgt_corpus.append(text)

    pairs = []
    for i in range(len(src_corpus)):
        if (len(src_corpus[i].split()) > 0) and (len(src_corpus[i].split()) < 50):
            if (len(tgt_corpus[i].split()) > 0) and (len(tgt_corpus[i].split()) < 50):
                pairs.append([src_corpus[i], tgt_corpus[i]])

    return pairs

def build_vocab():
    src_vocab, tgt_vocab = OrderedDict(), OrderedDict()
    
    with open("./data/en_de/vocab.50K.en.txt", "r") as f:
        for line in f:
            word = (line.replace("\n", "")).lower()
            src_vocab[word] = 1
            
    with open("./data/en_de/vocab.50K.de.txt", "r") as f:
        for line in f:
            word = (line.replace("\n", "")).lower()
            tgt_vocab[word] = 1
            
    return src_vocab, tgt_vocab


class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.pairs = read_data(src, tgt)

    def __getitem__(self, index):
        return self.pairs[index][0], self.pairs[index][1]

    def __len__(self):
        return len(self.pairs)

class TestDataset(Dataset):
    def __init__(self, src, tgt):
        self.pairs = read_test_data(src, tgt)

    def __getitem__(self, index):
        return self.pairs[index][0], self.pairs[index][1]

    def __len__(self):
        return len(self.pairs)

if __name__ == "__main__":
    special_symbols = ["<pad>"]
    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

    src_, tgt_ = build_vocab()
    src_vocab = vocab(src_, specials=special_symbols, special_first=True)
    tgt_vocab = vocab(tgt_, specials=special_symbols, special_first=True)
    src_vocab.set_default_index(UNK_IDX)
    tgt_vocab.set_default_index(UNK_IDX)

    # torch.save(src_vocab, "eng_vocab.pth")
    # torch.save(tgt_vocab, "ger_vocab.pth")

    tokenizer = lambda x: x.lower().split()
    src_pipeline = lambda x: src_vocab(tokenizer(x))
    tgt_pipeline = lambda x: tgt_vocab(tokenizer(x))
    # train_data = TranslationDataset("en", "de")
    # trainloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, num_workers=8)
    test_data = TestDataset("en", "de")
    testloader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True, num_workers=8)
    # torch.save(trainloader, "trainloader.pkl")
    torch.save(testloader, "testloader.pkl")
