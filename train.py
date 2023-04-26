import os
import sys
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from preprocess import TranslationDataset, TestDataset
from global_partials import *
from local_partials import *
from monotonic_partials import *
from predictive_partials import *
from model import *
from utils import *

torch.manual_seed(0)

if sys.argv[1] == "gpu":
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 32
EPOCHS = 10
NUM_LAYERS = 4
HIDDEN_SIZE = 1000
DROPOUT = 0.2
MAX_NORM = 5
PATH = sys.argv[2]+"_"+sys.argv[3] + "_attention.pt"

PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

src_vocab = torch.load("eng_vocab.pth")
tgt_vocab = torch.load("ger_vocab.pth")
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

tokenizer = lambda x: x.lower().split()
src_pipeline = lambda x: src_vocab(tokenizer(x))
tgt_pipeline = lambda x: tgt_vocab(tokenizer(x))

dataloader = torch.load("trainloader.pkl")
# dataloader = torch.load("testloader.pkl")

if sys.argv[3] == "global":
    if sys.argv[2] == "concat":
        attention = ConcatAttention(hidden_size=HIDDEN_SIZE)
    elif sys.argv[2] == "dot":
        attention = DotAttention(hidden_size=HIDDEN_SIZE)
    elif sys.argv[2] == "general":
        attention = GeneralAttention(hidden_size=HIDDEN_SIZE)
    elif sys.argv[2] == "location":
        attention = LocationAttention(hidden_size=HIDDEN_SIZE)
    encoder = Encoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    decoder = Decoder(attention=attention, vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

elif sys.argv[3] == "monotonic":
    if sys.argv[2] == "concat":
        attention = MonotonicConcatAttention(hidden_size=HIDDEN_SIZE)
    elif sys.argv[2] == "dot":
        attention = MonotonicDotAttention(hidden_size=HIDDEN_SIZE)
    elif sys.argv[2] == "general":
        attention = MonotonicGeneralAttention(hidden_size=HIDDEN_SIZE)
    encoder = LocalEncoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    decoder = LocalDecoder(attention=attention, vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    model = LocalSeq2Seq(encoder, decoder, device).to(device)

elif sys.argv[3] == "predictive":
    if sys.argv[2] == "concat":
        attention = PredictiveConcatAttention(hidden_size=HIDDEN_SIZE)
    elif sys.argv[2] == "dot":
        attention = PredictiveDotAttention(hidden_size=HIDDEN_SIZE)
    elif sys.argv[2] == "general":
        attention = PredictiveGeneralAttention(hidden_size=HIDDEN_SIZE)
    encoder = LocalEncoder(vocab_size=src_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    decoder = LocalDecoder(attention=attention, vocab_size=tgt_vocab_size, hidden_size=HIDDEN_SIZE, pad_token=PAD_IDX, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    model = LocalSeq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

try:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1
    print_loss = checkpoint["loss"]
except:
    model.apply(init_weights)
    epoch = 0
    print_loss = 0
    
print(f'Resume training at {epoch}!')

start_time = time.time()
for e in range(epoch, EPOCHS):
    for i, pairs in enumerate(dataloader):
        src, tgt, src_len = make_variables(pairs[0], pairs[1], device, src_pipeline, tgt_pipeline, PAD_IDX, SOS_IDX, EOS_IDX, True)
        mask = create_mask(src, PAD_IDX, device)
        pred = model(src, src_len, mask, tgt, False)
        
        pred_dim = pred.shape[-1]
        loss = criterion(pred[:-1].reshape(-1, pred_dim), tgt[1:].reshape(-1))
        print_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step()

        if i==0:
            print_loss = round(print_loss, 5)
            print("Time Elapsed: " + str(int(time.time() - start_time)) + " sec", end=" || ")
            print("Current Epoch: " + str(e) + " / " + str(EPOCHS), end=" || Iter: ")
            print(str(i)+ " / " + str(len(dataloader)))
            
        elif (i % 500 == 0):
            print_loss /= 500
            print_loss = round(print_loss, 5)
            print("Time Elapsed: " + str(int(time.time() - start_time)) + " sec", end=" || ")
            print("Current Epoch: " + str(e) + " / " + str(EPOCHS), end=" || Iter: ")
            print(str(i)+ " / " + str(len(dataloader)))
            print("Loss: " + str(print_loss), end=" || ")
            print("Perplexity: " + str(math.exp(print_loss)))
            print_loss = 0

    torch.save({
        "epoch":e,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        'loss': print_loss
    }, PATH)
    