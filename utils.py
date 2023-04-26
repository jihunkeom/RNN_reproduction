import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)

def create_mask(sentence, pad_token, device):
    mask = (sentence == pad_token).to(device)
    return mask

def make_variables(source, target, device, src_pipeline, tgt_pipeline, PAD_IDX=0, SOS_IDX=2, EOS_IDX=3, reverse=True):
    batch_size = len(source)
    src_text_list, tgt_text_list = [], []
    SOS = torch.tensor([SOS_IDX], dtype=torch.int64)
    EOS = torch.tensor([EOS_IDX], dtype=torch.int64)

    for i in range(batch_size):
        if reverse:
            src_text = torch.tensor(src_pipeline(source[i])[::-1], dtype=torch.int64)
        else:
            src_text = torch.tensor(src_pipeline(source[i]), dtype=torch.int64)
        tgt_text = torch.tensor(tgt_pipeline(target[i]), dtype=torch.int64)
        tgt_text = torch.cat((SOS, tgt_text, EOS))

        src_text_list.append([src_text, len(src_text)])
        tgt_text_list.append([tgt_text, len(src_text)])

    src_text_list = sorted(src_text_list, key = lambda x: -x[1])
    tgt_text_list = sorted(tgt_text_list, key = lambda x: -x[1])

    src_len = [a[1] for a in src_text_list]
    src_text_list = [a[0] for a in src_text_list]
    tgt_text_list = [a[0] for a in tgt_text_list]

    src = pad_sequence(src_text_list, batch_first=False, padding_value=PAD_IDX).to(device)
    tgt = pad_sequence(tgt_text_list, batch_first=False, padding_value=PAD_IDX).to(device)

    src_len = torch.tensor(src_len, dtype=torch.int64)

    return src, tgt, src_len