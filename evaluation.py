import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score
from torch.utils.data import Dataset, DataLoader

from preprocess import TranslationDataset, TestDataset
from global_partials import *
from local_partials import *
from monotonic_partials import *
from predictive_partials import *
from model import *
from utils import *

def translate(sentence, tgt, model, device, vocab, max_len=20, PAD_IDX=0):
    model.eval()

    src = torch.tensor(src_pipeline(sentence)[::-1], dtype=torch.int64).to(device)
    src_len = torch.tensor([len(src)])
    src = src.unsqueeze(1)
    mask = create_mask(src, PAD_IDX, device)

    # tgt = torch.tensor(tgt_pipeline(tgt), dtype=torch.int64).to(device)
    # tgt = tgt.unsqueeze(1)

    with torch.no_grad():
        generated = model.translate(src, src_len, mask)
        # pred = model(src, src_len, mask, tgt)
        # pred_dim = pred.shape[-1]
        # loss = criterion(pred[:-1].reshape(-1, pred_dim), tgt[1:].reshape(-1))

    generated = vocab.lookup_tokens(generated)
    
    return generated[:-1]#, loss.item()


def evaluate_by_one(model, dataloader, device, vocab, max_len=20, PAD_IDX=0):
    model.eval()

    outputs = []
    gold_label = []

    # print_loss = 0

    with torch.no_grad():
        for i, pairs in enumerate(dataloader):
            src, tgt = pairs[0][0], pairs[1][0]
            output = translate(src, tgt, model, device, vocab, max_len)
            outputs.append(output)
            gold_label.append([tgt.split()])
            # print_loss += loss
            # ppl = math.exp(print_loss / (i+1))

            print(tgt.split())
            print(output)
            print(f"BLEU 1 : {bleu_score(outputs, gold_label, 1, [1])} || BLEU 2 : {bleu_score(outputs, gold_label, 4, [0.45, 0.3, 0.2, 0.05])}")
            print("-"*30)

    return bleu_score(outputs, gold_label, 1, [1]), bleu_score(outputs, gold_label, 4, [0.45, 0.3, 0.2, 0.05])

if __name__ == "__main__":

    device = torch.device("cuda:1")
    # device = torch.device("cpu")

    BATCH_SIZE = 1
    NUM_LAYERS = 4
    HIDDEN_SIZE = 1000
    DROPOUT = 0.2

    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
    src_vocab = torch.load("eng_vocab.pth")
    tgt_vocab = torch.load("ger_vocab.pth")

    tokenizer = lambda x: x.lower().split()
    src_pipeline = lambda x: src_vocab(tokenizer(x))
    tgt_pipeline = lambda x: tgt_vocab(tokenizer(x))

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    testloader = torch.load("testloader.pkl")

    if sys.argv[1] == "dot":
        if sys.argv[2] == "global":
            attention = DotAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("dot_global_attention.pt", map_location=device)
        elif sys.argv[2] == "monotonic":
            attention = MonotonicDotAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("dot_monotonic_attention.pt", map_location=device)
        elif sys.argv[2] == "predictive":
            attention = PredictiveDotAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("dot_predictive_attention.pt", map_location=device)

    if sys.argv[1] == "location":
        if sys.argv[2] == "global":
            attention = LocationAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("location_global_attention.pt", map_location=device)

    if sys.argv[1] == "concat":
        if sys.argv[2] == "global":
            attention = ConcatAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("concat_global_attention.pt", map_location=device)
        elif sys.argv[2] == "monotonic":
            attention = MonotonicConcatAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("concat_monotonic_attention.pt", map_location=device)
        elif sys.argv[2] == "predictive":
            attention = PredictiveConcatAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("concat_predictive_attention.pt", map_location=device)

    if sys.argv[1] == "general":
        if sys.argv[2] == "global":
            attention = GeneralAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("general_global_attention.pt", map_location=device)
        elif sys.argv[2] == "monotonic":
            attention = MonotonicGeneralAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("general_monotonic_attention.pt", map_location=device)
        elif sys.argv[2] == "predictive":
            attention = PredictiveGeneralAttention(hidden_size=HIDDEN_SIZE)
            checkpoint = torch.load("general_predictive_attention.pt", map_location=device)

    if sys.argv[2] == "global":
        encoder = Encoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        decoder = Decoder(attention=attention, vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
    else:
        encoder = LocalEncoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        decoder = LocalDecoder(attention=attention, vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        model = LocalSeq2Seq(encoder, decoder, device).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"model loaded at {checkpoint['epoch']+1} epoch")

    bleu, bleu2 = evaluate_by_one(model, testloader, device, tgt_vocab, max_len=20, PAD_IDX=PAD_IDX)
    print(bleu, bleu2)
