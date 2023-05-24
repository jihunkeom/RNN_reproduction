import random
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, src_mask, tgt, teacher_forcing=False):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        vocab_size = self.decoder.vocab_size

        enc_output, hidden, cell = self.encoder(src, src_len)
        dec_outputs = torch.zeros((tgt_len, batch_size, vocab_size)).to(self.device)

        tgt_input = tgt[0, :]
        attentional_vector = torch.zeros(1, self.encoder.batch_size, self.encoder.hidden_size).to(self.device)

        for t in range(tgt_len-1):
            output, attentional_vector, hidden, cell = self.decoder(tgt_input, hidden, cell, enc_output, src_mask, attentional_vector)
            dec_outputs[t] = output

            if teacher_forcing > 0:
                if random.random() < teacher_forcing_ratio:
                    tgt_input = tgt[t, :]
                else:
                    tgt_input = output.argmax(2).squeeze(0)
            
            else:
                tgt_input = tgt[t, :]

        return dec_outputs


    def translate(self, src, src_len, mask, SOS_IDX=2, EOS_IDX=3, max_len=20):
        generated = []
        enc_output, hidden, cell = self.encoder(src, src_len)
        tgt_input = torch.tensor([SOS_IDX], dtype=torch.int64).to(self.device)
        attentional_vector = torch.zeros(1, 1, self.encoder.hidden_size).to(self.device)

        for t in range(max_len):
            output, attentional_vector, hidden, cell = self.decoder(tgt_input, hidden, cell, enc_output, mask, attentional_vector)
            pred = output.argmax(2).item()
            generated.append(pred)
            if pred == EOS_IDX:
                break
            tgt_input = torch.tensor([pred], dtype=torch.int64).to(self.device)

        return generated
