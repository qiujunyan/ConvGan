import math
import os
import time
from torch import optim
import torch as t
from torch import nn
from codes.transformer import *
from codes.train import Trainer


class PreTrainer(Trainer):
  def __init__(self, data_dir="./data/mutual/dev"):
    Trainer.__init__(self, data_dir)
    self.epoch_num = 1000
    self.g_lr0 = 1e-4
    self.reg_rate =1e-4
    self.generator = Transformer(self.embedding, d_ff=self.d_ff, num_heads=self.num_heads,
                                 num_layers=self.g_num_layers, dropout=self.g_dropout,
                                 pad_id=self.special_tokens[self.pad_tok]).to(device=self.device)
    self.g_optim = optim.Adam(self.generator.parameters(), lr=self.g_lr0)
    self.init_params()

  def forward(self):
    print("*" * 20 + "device: {}".format(self.device) + "*" * 20)
    self.total_batch = math.ceil(self.data_size / self.batch_size)
    self.total_batch = 1
    print("total batch: {}".format(self.total_batch))
    # self.load_generator("./pretrain/generator/model/model-20-0")

    self.model_dir = os.path.join("./pretrain/generator/model/", "model-{}"
                                  .format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    self.record_file = os.path.join(self.model_dir, "record.txt")
    self.record_params()
    for epoch in range(self.epoch_num):
      self.shuffle_data()
      self.train_one_epoch(epoch)

  def train_one_epoch(self, epoch):
    with open(self.record_file, "a") as f:
      f.write("\n")
    for batch in range(self.total_batch):
      self.g_optim.zero_grad()
      src, tgt, src_mask, tgt_mask, ans_lens = self.get_batch(batch)
      self.lr = self.adjust_lr(self.g_lr0, self.g_optim,
                               epoch, self.epoch_num, min_lr=None,
                               warmup=30, warmup_lr=1e-6, power=10)
      logits = self.generator(src, tgt, src_mask, tgt_mask)
      gen_sents = t.argmax(logits, -1)
      loss = self.get_celoss(tgt[:, 1:], logits[:, :-1], self.generator.parameters())

      # optimization
      self.g_optim.zero_grad()
      loss.backward()
      # some error occurs when device is gpu, gradient of
      # embedding of padding index goes nan.
      self.embedding.weight.grad[self.special_tokens[self.pad_tok]] = 0
      self.g_optim.step()

      if batch == 0 and epoch % 10 == 0:
        acc = self.get_accuracy(gen_sents[:, :-1], tgt[:, 1:])
        self.log_record(epoch, batch, loss, gen_sents, tgt,
                        g_acc=acc, ans_lens=ans_lens)
        if epoch % 10 == 0:
          self.model_save(epoch, batch)
        print("\n")

  def init_params(self):
    for params in self.generator.parameters():
      try:
        nn.init.xavier_normal_(params)
      except ValueError:
        pass

  def get_celoss(self, labels, logits, parameters=None):
    reg_term = 0
    if parameters is not None:
      for param in parameters:
        reg_term += t.norm(param)

    def _get_onehot(index, res):
      ret = t.eye(self.vocab_size, device=index.device)[index]
      ret += (1 - ret) * (res / (self.embed_dim - 1)) - ret * res
      return ret

    index1, index2 = t.arange(0, labels.size(1)), labels[0]
    tmp = logits[0][index1, index2]
    is_target = (labels == 0).float()
    labels = _get_onehot(labels, res=0.1)
    loss = -t.sum(labels * t.log(logits + 1e-8)) / t.sum(is_target)
    return loss + self.reg_rate * reg_term


if __name__ == "__main__":
  trainer = PreTrainer()
  trainer()
