import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

class LocationAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_output, src_mask):
        src_mask = src_mask.unsqueeze(2)
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(0)
        decoder_hidden = decoder_hidden.expand_as(encoder_output).contiguous()

        att_weights = self.dense(decoder_hidden)
        att_weights.masked_fill_(src_mask!=0, -1e10)
        att_weights = F.softmax(att_weights, dim=0)

        context_vector = torch.sum(att_weights * encoder_output, axis=0)
        context_vector = context_vector.unsqueeze(0)

        return context_vector

class DotAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

    def forward(self, decoder_hidden, encoder_output, src_mask):
        src_mask = torch.transpose(src_mask, 0, 1).contiguous().unsqueeze(1)
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(0)
        
        att_weights = torch.bmm(decoder_hidden.permute(1, 0, 2).contiguous(), encoder_output.permute(1, 2, 0).contiguous())
        att_weights.masked_fill_(src_mask!=0, -1e10)
        att_weights = F.softmax(att_weights, dim=-1)
        
        context_vector = torch.sum(att_weights.permute(2, 0, 1).contiguous()*encoder_output, axis=0)
        context_vector = context_vector.unsqueeze(0)

        return context_vector


class ConcatAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_output, src_mask):
        src_mask = src_mask.unsqueeze(-1)
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(0)
        decoder_hidden = decoder_hidden.expand_as(encoder_output).contiguous()

        cat = torch.cat((decoder_hidden, encoder_output), axis=-1)
        att_weights = self.v(torch.tanh(self.dense(cat)))
        att_weights.masked_fill_(src_mask!=0, -1e10)
        att_weights = F.softmax(att_weights, dim=0)
        
        context_vector = torch.sum(att_weights * encoder_output, axis=0)
        context_vector = context_vector.unsqueeze(0)

        return context_vector

class GeneralAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        
    
    def forward(self, decoder_hidden, encoder_output, src_mask):
        src_mask = src_mask.unsqueeze(-1)
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(0)

        decoder_hidden = self.dense(decoder_hidden)
        att_weights = torch.bmm(decoder_hidden.permute(1, 0, 2).contiguous(), encoder_output.permute(1, 2, 0).contiguous())
        att_weights = att_weights.permute(2, 0, 1)
        att_weights.masked_fill_(src_mask!=0, -1e10)
        att_weights = F.softmax(att_weights, dim=0)

        context_vector = torch.sum(att_weights * encoder_output, axis=0)
        context_vector = context_vector.unsqueeze(0)

        # 가중평균 말고 행렬곱으로 해볼수도 있을듯 (general 말고 다른 방식에도 적용 가능) (속도는 좀 더 빠름)
        # context_vector = torch.bmm(att_weights.permute(1, 2, 0), encoder_output.permute(1, 0, 2))
        # context_vector = context_vector.permute(1, 0, 2)

        return context_vector
        

class Encoder(nn.Module):
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

class Decoder(nn.Module):
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

    def forward(self, tgt, hidden, cell, enc_output, src_mask, attentional_vector):
        tgt = tgt.unsqueeze(0)
        embedded = self.embedding(tgt)
        # print(embedded.shape, tgt.shape, attentional_vector.shape)
        input_feeding = torch.cat((embedded, attentional_vector), axis=-1)
        output, (hidden, cell) = self.lstm(input_feeding, (hidden, cell))
        
        context_vector = self.attention(hidden, enc_output, src_mask)
        
        attentional_vector = torch.tanh(self.dense1(torch.cat((output, context_vector), axis=-1))) #input feeding
        output = self.dense2(attentional_vector)
        return output, attentional_vector, hidden, cell
