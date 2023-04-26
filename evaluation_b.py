import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score
from torch.utils.data import Dataset, DataLoader

from preprocess import TranslationDataset, TestDataset
from baseline import *
from utils import *

def translate(sentence, tgt, model, device, vocab, max_len, PAD_IDX, reverse):
    model.eval()
    if reverse:
        src = torch.tensor(src_pipeline(sentence)[::-1], dtype=torch.int64).to(device)
    else:
        src = torch.tensor(src_pipeline(sentence), dtype=torch.int64).to(device)
    src_len = torch.tensor([len(src)])
    src = src.unsqueeze(1)
    mask = create_mask(src, PAD_IDX, device)

    with torch.no_grad():
        generated = model.translate(src, src_len, mask)

    generated = vocab.lookup_tokens(generated)
    
    return generated[:-1]


def evaluate_by_one(model, dataloader, device, vocab, max_len=20, PAD_IDX=0, reverse=True):
    model.eval()
    outputs = []
    gold_label = []

    with torch.no_grad():
        for i, pairs in enumerate(dataloader):
            src, tgt = pairs[0][0], pairs[1][0]
            output = translate(src, tgt, model, device, vocab, max_len, PAD_IDX, reverse)
            outputs.append(output)
            gold_label.append([tgt.split()])
            
            print(tgt.split())
            print(output)
            print(f"BLEU 1 : {bleu_score(outputs, gold_label, 1, [1])} || BLEU 2 : {bleu_score(outputs, gold_label, 4, [0.45, 0.3, 0.2, 0.05])} ")
            print("-"*30)

    return bleu_score(outputs, gold_label, 1, [1]), bleu_score(outputs, gold_label, 4, [0.45, 0.3, 0.2, 0.05])

if __name__ == "__main__":
    device = torch.device("cuda:1")
    # device = torch.device("cpu")

    BATCH_SIZE = 1
    NUM_LAYERS = 4
    HIDDEN_SIZE = 1000
    DROPOUT = 0

    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
    src_vocab = torch.load("eng_vocab.pth")
    tgt_vocab = torch.load("ger_vocab.pth")

    tokenizer = lambda x: x.lower().split()
    src_pipeline = lambda x: src_vocab(tokenizer(x))
    tgt_pipeline = lambda x: tgt_vocab(tokenizer(x))

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    testloader = torch.load("testloader.pkl")

    checkpoint = torch.load(f"baseline_{sys.argv[1]}.pt", map_location=device)

    if sys.argv[1] == "1":
        encoder = BaselineEncoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=0).to(device)
        decoder = BaselineDecoder(vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=0).to(device)
        model = Baseline(encoder, decoder, device).to(device)
    elif sys.argv[1] == "2":
        encoder = BaselineEncoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=0).to(device)
        decoder = BaselineDecoder(vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=0).to(device)
        model = Baseline(encoder, decoder, device).to(device)
    elif sys.argv[1] == "3":
        encoder = BaselineEncoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        decoder = BaselineDecoder(vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        model = Baseline(encoder, decoder, device).to(device)
    elif sys.argv[1] == "4":
        encoder = BaselineEncoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        decoder = BaselineDecoder4(vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        model = Baseline4(encoder, decoder, device).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"model loaded at {checkpoint['epoch']+1} epoch")

    bleu, bleu2 = evaluate_by_one(model, testloader, device, tgt_vocab, max_len=20, PAD_IDX=PAD_IDX, reverse=True)
    print(bleu, bleu2)