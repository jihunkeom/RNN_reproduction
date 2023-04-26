import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token, batch_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, 
        batch_first=False, dropout=0, bidirectional=True)

        self.dense = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded)

        hidden_cat = torch.cat((hidden[0::2, :, :], hidden[1::2, :, :]), axis=2)
        cell_cat = torch.cat((cell[0::2, :, :], cell[1::2, :, :]), axis=2)
        hidden = torch.tanh(self.dense(hidden_cat))
        cell = torch.tanh(self.dense(cell_cat))

        return output, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.dense = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, decoder_hidden, encoder_output):
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(1)
        encoder_output = self.dense(encoder_output).permute(1, 2, 0)

        energy = torch.bmm(decoder_hidden, encoder_output)
        weights = F.softmax(energy, dim=-1)

        attention_vector = torch.bmm(weights, encoder_output.permute(0, 2, 1)).permute(1, 0, 2)
        return attention_vector

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token, batch_size, attention, num_layers=1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, 
        batch_first=False, dropout=0, bidirectional=False)
        
        self.dense1 = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        self.attention = attention

    def forward(self, input, hidden, cell, encoder_output):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        attention_vec = self.attention(hidden, encoder_output)
        concat_vec = torch.cat((attention_vec, output), axis=-1)
        out = self.dense1(concat_vec.squeeze(0))
        
        pred = self.dense2(out)
        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        encoder_output, hidden, cell = self.encoder(src)

        tgt_input = tgt[0, :]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(tgt_input, hidden, cell, encoder_output)
            outputs[t] = output

            top1 = output.argmax(1)
            if random.random() < teacher_forcing_ratio:
                tgt_input = tgt[t, :]
            else:
                tgt_input = top1

        return outputs

