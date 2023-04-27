import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

class LocalEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token, batch_size, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
        batch_first=False, dropout=dropout, bidirectional=False)

    def forward(self, src, input_len):
        embedded = self.embedding(src)
        packed_embedded = pack_padded_sequence(embedded, input_len.to("cpu"), batch_first=False, enforce_sorted=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=False, padding_value=0)
        return output, hidden, cell

class LocalDecoder(nn.Module):
    def __init__(self, attention, vocab_size, hidden_size, pad_token, batch_size, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=num_layers,
        batch_first=False, dropout=dropout, bidirectional=False)
        
        self.dense1 = nn.Linear(in_features=hidden_size*2, out_features=hidden_size, bias=False)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False)

    def forward(self, tgt, hidden, cell, enc_output, src_mask, attentional_vector, t):
        tgt = tgt.unsqueeze(0)
        embedded = self.embedding(tgt)
        input_feeding = torch.cat((embedded, attentional_vector), axis=-1)
        output, (hidden, cell) = self.lstm(input_feeding, (hidden, cell))
        
        context_vector = self.attention(hidden, enc_output, src_mask, t)

        attentional_vector = torch.tanh(self.dense1(torch.cat((output, context_vector), axis=-1))) #input feeding
        output = self.dense2(attentional_vector)
        return output, attentional_vector, hidden, cell