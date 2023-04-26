import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BaselineEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
        batch_first=False, dropout=dropout, bidirectional=False)

    def forward(self, src, input_len):
        embedded = self.embedding(src)
        packed_embedded = pack_padded_sequence(embedded, input_len.to("cpu"), batch_first=False, enforce_sorted=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=False, padding_value=0)
        return output, hidden, cell

class BaselineDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
        batch_first=False, dropout=dropout, bidirectional=False)
        
        self.dense = nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False)

    def forward(self, tgt, hidden, cell):
        tgt_embedded = self.embedding(tgt)
        output, (hidden, cell) = self.lstm(tgt_embedded, (hidden, cell))
        output = self.dense(output)
        return output, hidden, cell

class BaselineDecoder4(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=False, dropout=dropout, bidirectional=False)
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.dense2 = nn.Linear(in_features=hidden_size*2, out_features=vocab_size, bias=False)

    def forward(self, tgt, hidden, cell, enc_output, src_mask):
        tgt_embedded = self.embedding(tgt)
        dec_output, (hidden, cell) = self.lstm(tgt_embedded, (hidden, cell))
        
        src_mask = src_mask.unsqueeze(2)
        att_weights = self.dense1(dec_output)
        context_vector = torch.zeros_like(dec_output)
        for i in range(dec_output.shape[0]):
            tmp = att_weights[i].expand_as(enc_output).contiguous()
            tmp.masked_fill_(src_mask!=0, -1e10)
            weight = F.softmax(tmp, dim=0)
            context_vector[i] = torch.sum(weight * enc_output, axis=0)

        output = self.dense2(torch.cat((dec_output, context_vector), axis=-1))
        return output, hidden, cell

class Baseline(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, tgt):
        enc_output, hidden, cell = self.encoder(src, src_len)
        output, dec_hidden, dec_cell = self.decoder(tgt, hidden, cell)
        return output

    def translate(self, src, src_len, mask, SOS_IDX=2, EOS_IDX=3, max_len=20):
        generated = []
        enc_output, hidden, cell = self.encoder(src, src_len)
        tgt_input = torch.tensor([SOS_IDX], dtype=torch.int64).unsqueeze(-1).to(self.device)

        for t in range(max_len):
            output, hidden, cell = self.decoder(tgt_input, hidden, cell)
            tgt_input = output.argmax(2)
            pred = tgt_input.item()
            generated.append(pred)
            if pred == EOS_IDX:
                break

        return generated

class Baseline4(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, tgt, src_mask):
        enc_output, hidden, cell = self.encoder(src, src_len)
        output, dec_hidden, dec_cell = self.decoder(tgt, hidden, cell, enc_output, src_mask)
        return output

    def translate(self, src, src_len, mask, SOS_IDX=2, EOS_IDX=3, max_len=20):
        generated = []
        enc_output, hidden, cell = self.encoder(src, src_len)
        tgt_input = torch.tensor([SOS_IDX], dtype=torch.int64).unsqueeze(-1).to(self.device)

        for t in range(max_len):
            output, hidden, cell = self.decoder(tgt_input, hidden, cell, enc_output, mask)
            tgt_input = output.argmax(2)
            pred = tgt_input.item()
            generated.append(pred)
            if pred == EOS_IDX:
                break

        return generated
