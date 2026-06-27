import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(
            emb_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.lin = nn.Linear(2 * hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)
        return scores

    def test(self, sents_tensor, lengths, _):
        logits = self.forward(sents_tensor, lengths)
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids
