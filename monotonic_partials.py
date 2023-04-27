import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


class MonotonicDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

    def forward(self, decoder_hidden, encoder_output, src_mask, t):
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(0)
        src_len = encoder_output.shape[0]

        if (t-10) > 0:
            if (t+10) > src_len:
                encoder_output = encoder_output[t-10:]
                src_mask = src_mask[t-10:]
            else:
                encoder_output = encoder_output[t-10:t+10]
                src_mask = src_mask[t-10:t+10]
        else:
            if (t+10) < src_len:
                encoder_output = encoder_output[:t+10]
                src_mask = src_mask[:t+10]

        src_mask = torch.transpose(src_mask, 0, 1).unsqueeze(1)
        
        att_weights = torch.bmm(decoder_hidden.permute(1, 0, 2).contiguous(), encoder_output.permute(1, 2, 0).contiguous())
        att_weights.masked_fill_(src_mask!=0, -1e10)
        att_weights = F.softmax(att_weights, dim=-1)
        
        context_vector = torch.sum(att_weights.permute(2, 0, 1).contiguous()*encoder_output, axis=0)
        context_vector = context_vector.unsqueeze(0)

        return context_vector


class MonotonicConcatAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_output, src_mask, t):
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(0)
        src_len = encoder_output.shape[0]
        
        if (t-10) > 0:
            if (t+10) > src_len:
                encoder_output = encoder_output[t-10:]
                src_mask = src_mask[t-10:]
            else:
                encoder_output = encoder_output[t-10:t+10]
                src_mask = src_mask[t-10:t+10]
        else:
            if (t+10) < src_len:
                encoder_output = encoder_output[:t+10]
                src_mask = src_mask[:t+10]

        src_mask = src_mask.unsqueeze(-1)
        decoder_hidden = decoder_hidden.expand_as(encoder_output).contiguous()

        cat = torch.cat((decoder_hidden, encoder_output), axis=-1)
        att_weights = self.v(torch.tanh(self.dense(cat)))
        att_weights.masked_fill_(src_mask!=0, -1e10)#-float("inf"))
        att_weights = F.softmax(att_weights, dim=0)
        
        context_vector = torch.sum(att_weights * encoder_output, axis=0)
        context_vector = context_vector.unsqueeze(0)

        return context_vector

class MonotonicGeneralAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, decoder_hidden, encoder_output, src_mask, t):
        decoder_hidden = decoder_hidden[-1, :, :].unsqueeze(0)
        src_len = encoder_output.shape[0]
        
        if (t-10) > 0:
            if (t+10) > src_len:
                encoder_output_ = encoder_output[t-10:]
                src_mask_ = src_mask[t-10:]
            else:
                encoder_output_ = encoder_output[t-10:t+10]
                src_mask_ = src_mask[t-10:t+10]
        else:
            if (t+10) < src_len:
                encoder_output_ = encoder_output[:t+10]
                src_mask_ = src_mask[:t+10]
            else:
                encoder_output_ = encoder_output
                src_mask_ = src_mask

        src_mask_ = src_mask_.unsqueeze(-1)
        decoder_hidden = self.dense(decoder_hidden)
        att_weights = torch.bmm(decoder_hidden.permute(1, 0, 2).contiguous(), encoder_output_.permute(1, 2, 0).contiguous())
        att_weights = att_weights.permute(2, 0, 1).contiguous()
        att_weights.masked_fill_(src_mask_ != 0, -1e10)
        att_weights = F.softmax(att_weights, dim=0)

        context_vector = torch.sum(att_weights * encoder_output_, axis=0)
        context_vector = context_vector.unsqueeze(0)

        return context_vector

