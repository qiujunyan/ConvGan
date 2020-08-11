import math
import os
import time

import torch as t
from torch import nn

from codes.train import Trainer


class PreTrainer(Trainer):
  def __init__(self, data_dir="./data/mutual/dev"):
    Trainer.__init__(self, data_dir)
    self.epoch_num = 100
    self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.special_tokens[self.pad_tok])
    self.g_lr0 = 1e-1
    self.reg_rate = 0

  def forward(self):
    print("*" * 20 + "device: {}".format(self.device) + "*" * 20)
    self.total_batch = math.ceil(self.data_size / self.batch_size)
    self.total_batch = 1
    print("total batch: {}".format(self.total_batch))
    # self.load_generator("./pretrain/generator/model/model-2020-08-07_23-00-42/model-3-10")

    self.model_dir = os.path.join("./pretrain/generator/model/", "model-{}"
                                  .format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    self.record_file = os.path.join(self.model_dir, "record.txt")
    self.record_params()
    for epoch in range(self.epoch_num):
      self.train_one_epoch(epoch)

  def train_one_epoch(self, epoch):
    with open(self.record_file, "a") as f:
      f.write("\n")
    for batch in range(self.total_batch):
      self.g_optim.zero_grad()
      answer, dialog, src_mask, tgt_mask, ans_lens = self.get_batch(batch)
      self.lr = self.adjust_lr(self.g_lr0, self.g_optim,
                               epoch * self.total_batch + batch,
                               self.epoch_num * self.total_batch,
                               min_lr=1e-4, warmup=0, warmup_lr=1e-4)
      logits = self.generator(src=dialog, tgt=answer,
                              src_mask=src_mask, tgt_mask=tgt_mask,
                              is_pretraining=True)
      gen_sents = t.argmax(logits, -1)
      loss = self.get_celoss(answer, logits, self.generator.parameters())

      # optimization
      self.g_optim.zero_grad()
      loss.backward()
      # some error occurs when device is gpu, gradient of
      # embedding of padding index goes nan.
      self.embedding.weight.grad[self.special_tokens[self.pad_tok]] = 0
      self.g_optim.step()

      if epoch % 5 == 0:
        acc = self.get_accuracy(gen_sents, answer[:, 1:])
        self.log_record(epoch, batch, loss, gen_sents, answer,
                        g_acc=acc, ans_lens=ans_lens)
        self.model_save(epoch, batch)
        print("\n")

  def get_celoss(self, labels, logits, parameters=None, smooth=False):
    reg_term = 0
    if parameters is not None:
      for param in parameters:
        reg_term += t.norm(param)

    if smooth:
      def _get_onehot(index, res=0):
        ret = t.eye(self.vocab_size, device=index.device)[index]
        ret += (1 - ret) * (res / (self.embed_dim - 1)) - ret * res
        return ret

      labels = _get_onehot(labels[:, 1:], res=0.2)
      loss = -t.mean(labels * t.log(logits + 1e-8))
    else:
      loss = self.cross_entropy_loss(logits.permute(0, 2, 1), labels[:, 1:])
    loss = loss + self.reg_rate * reg_term
    return loss


if __name__ == "__main__":
  trainer = PreTrainer()
  trainer()
gen_sents
