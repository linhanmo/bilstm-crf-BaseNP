import os
from itertools import zip_longest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from bilstm import BiLSTM


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)
        self.transition = nn.Parameter(torch.ones(out_size, out_size) * 1 / out_size)

    def forward(self, sents_tensor, lengths):
        emission = self.bilstm(sents_tensor, lengths)
        batch_size, _, out_size = emission.size()
        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)
        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        start_id = tag2id["<start>"]
        end_id = tag2id["<end>"]
        pad_id = tag2id["<pad>"]
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        batch_size, max_len, _, _ = crf_scores.size()

        viterbi = torch.zeros(batch_size, max_len, tagset_size).to(device)
        backpointer = (torch.zeros(batch_size, max_len, tagset_size).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)

        for step in range(max_len):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                viterbi[:batch_size_t, step, :] = crf_scores[:batch_size_t, step, start_id, :]
                backpointer[:batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step - 1, :].unsqueeze(2) + crf_scores[:batch_size_t, step, :, :],
                    dim=1,
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        backpointer = backpointer.view(batch_size, -1)
        tagids = []
        tags_t = None
        for step in range(max_len - 1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == max_len - 1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)
                new_in_batch = torch.LongTensor([end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat([tags_t, new_in_batch], dim=0)
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            tags_t = backpointer[:batch_size_t].gather(dim=1, index=index.unsqueeze(1).long()).squeeze(1)
            tagids.append(tags_t.tolist())

        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad_id))
        return torch.Tensor(tagids).long()


BiLSTMCRF = BiLSTM_CRF
