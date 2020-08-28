import torch as t
import torch.nn as nn
from torch.utils.data import random_split
from codes.transformer import *
import pickle as pkl


class DataLoader(nn.Module):
  def __init__(self, file="test/data/train.pkl"):
    super(DataLoader, self).__init__()
    self.pad = 0
    self.bos = 1
    self.eos = 2

    self.data_size = 1000
    self.train_prob = 0.3
    self.vocab_size = 10000
    self.embed_dim = 512
    self.d_ff = 1024
    self.num_heads = 8
    self.num_layers = 6
    self.dropout = 0.1
    self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad)

    self.src_max_len = 300
    self.tgt_max_len = 32
    self.epoch_num = 100
    self.batch_size = 64
    self.total_batch = math.ceil(self.data_size / self.batch_size)

    # self.generate_data()
    self.src, self.tgt = self.load_data(file)

  def generate_data(self):
    def _pad_seq(max_len, lens):
      return t.cat(list(map(lambda i: t.cat([t.ones(1, dtype=t.int64) * self.bos,
                                             t.randint(0, self.vocab_size, [lens[i]]),
                                             t.ones(max_len - lens[i], dtype=t.int64) * self.pad,
                                             t.ones(1, dtype=t.int64) * self.eos], 0), range(lens.size(0)))), 0)

    src_seq_lens = t.randint(10, self.src_max_len, [self.data_size])
    tgt_seq_lens = t.randint(5, self.tgt_max_len, [self.data_size])

    src = _pad_seq(self.src_max_len, src_seq_lens)
    tgt = _pad_seq(self.tgt_max_len, tgt_seq_lens)
    neg_tgt = _pad_seq(self.tgt_max_len, tgt_seq_lens)

    train_src = src[:int(self.data_size * self.train_prob)]
    train_tgt = tgt[:int(self.data_size * self.train_prob)]
    train_neg_tgt = tgt[:int(self.data_size * self.train_prob)]
    dev_src = src[:int(self.data_size * (1 - self.train_prob))]
    dev_tgt = tgt[:int(self.data_size * (1 - self.train_prob))]

    with open("test/data/train.pkl", "wb") as f:
      pkl.dump([train_src, train_tgt], f)
    with open("test/data/dev.pkl", "wb") as f:
      pkl.dump([dev_src, dev_tgt], f)

  def load_data(self, file):
    with open(file, "rb") as f:
      src, tgt = pkl.load(f)

  def get_batch(self, index):
    start = index * self.batch_size
    end = min((index + 1) * self.batch_size, self.data_size)
    tgt = self.src[start:end, :]
    src = self.tgt[start:end, :]
    ans_lens = self.ans_lens[start:end]

    # get mask
    src_mask = (src == self.pad).to(src.device)
    tgt_mask = (tgt == self.pad).to(tgt.device)
    return src, tgt, src_mask, tgt_mask


class Discriminator(DataLoader):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.device = "cuda:0"
    self.transormer = Transformer(self.embedding, d_ff=self.d_ff, num_heads=self.num_heads,
                                  num_layers=self.num_layers, dropout=self.dropout,
                                  pad_id=self.pad).to(device=self.device)

  def forward(self):
    for epoch in range(self.epoch_num):
      self.train_one_epoch(epoch)
    logits = self.transormer.classify()

  def train_one_epoch(self, epoch):
    for batch in range(self.total_batch):
      loss = self.transormer()

  def get_batch(self, index):
    start = index * self.batch_size
    end = min((index + 1) * self.batch_size, self.data_size)
    tgt = self.tgt[start:end, :]
    src = self.src[start:end, :]

    # get mask
    src_mask = (src == self.pad).to(src.device)
    tgt_mask = (tgt == self.pad).to(tgt.device)
    return src, tgt, src_mask, tgt_mask


if __name__ == "__main__":
  dataloader = DataLoader()
