import os
import sys
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from preprocess import TranslationDataset, TestDataset
from baseline import *
from utils import *

torch.manual_seed(0)

device = torch.device("cuda:0")
# device = torch.device("cpu")
    
torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 32
EPOCHS = 10
NUM_LAYERS = 4
HIDDEN_SIZE = 1000
DROPOUT = 0.2
MAX_NORM = 5
PATH = f"baseline_{sys.argv[1]}.pt"

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

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

try:
    checkpoint = torch.load(PATH, map_location=device)
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
        #지금 reverse로 설정돼있는지 확인해보기 !
        src, tgt, src_len = make_variables(pairs[0], pairs[1], device, src_pipeline, tgt_pipeline, PAD_IDX, SOS_IDX, EOS_IDX, True)
        # pred = model(src, src_len, tgt)
        mask = create_mask(src, PAD_IDX, device)
        pred = model(src, src_len, tgt, mask)
        
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
    