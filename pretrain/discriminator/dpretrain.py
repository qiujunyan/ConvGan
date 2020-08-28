import os
import time

import numpy as np
import torch as t
from torch import nn
import math
from codes.parameters import Args
from codes.train import Trainer


class Pretrain(Trainer):
  def __init__(self):
    super(Pretrain, self).__init__(data_dir="data/mutual/train")
    self.epoch_num = 10
    self.d_lr0 = 1e-3
    self.batch_size = 32
    self.neg_datas = t.LongTensor(self.data_ids["wrong_ans"]).to(self.device)
    self.total_batch = math.ceil(self.data_size / self.batch_size)
    # self.total_batch = 1

  def forward(self):
    self.model_dir = os.path.join("./pretrain/discriminator/model", "model-{}".
                                  format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    self.load_discriminator("./pretrain/discriminator/model/model-2020-08-19_10-06-15/model-10-27")
    self.load_generator("./pretrain/generator/model/model-2020-08-14_11-36-59/model-250-0")
    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    self.record_file = os.path.join(self.model_dir, "log.txt")
    self.print_train_info()
    for epoch in range(self.epoch_num):
      self.train_one_epoch(epoch)

  def train_one_epoch(self, epoch):
    self.lr = self.adjust_lr(iter=epoch, max_iter=1000,
                             warmup=0, warmup_lr=1e-3,
                             power=10, min_lr=None)
    for batch in range(self.total_batch):
      self.d_optim.zero_grad()
      answer, neg_answer, dialog = self.get_batch(batch)
      loss, acc = self.discriminator(dialog, neg_answer, answer)

      loss.backward()
      self.d_optim.step()
    self.log_record(epoch, batch, loss, acc)
    if epoch % 10 == 0:
      self.model_save(epoch, batch)

  def get_batch(self, index):
    start = index * self.batch_size
    end = min((index + 1) * self.batch_size, self.data_size)
    answer = self.answer[start:end, :]
    neg_answer = self.neg_datas[start:end, :]
    dialog = self.dialog[start:end, :]

    neg_answer = self.generator(dialog, )
    return answer, neg_answer, dialog

  def adjust_lr(self, iter, max_iter, min_lr=None, warmup=None, warmup_lr=1e-3, power=5):
    if warmup is not None and iter < warmup:
      lr = warmup_lr
    else:
      lr = self.d_lr0 * ((1 - float(iter) / max_iter) ** power)
      if min_lr is not None:
        lr = max(min_lr, lr)
    self.d_optim.param_groups[0]["lr"] = lr
    return lr

  def model_save(self, epoch, batch):
    save_path = os.path.join(self.model_dir,
                             "model-{}-{}".format(epoch, batch))
    t.save({"epoch": epoch,
            "batch": batch,
            "discriminator_state_dict": self.discriminator.state_dict(),
            "lr": self.lr}, save_path)

  def log_record(self, epoch, batch, loss, acc):
    print("epoch {}, batch: {}, lr: {:.5}, loss: {:.3}, acc: {:.3}".
          format(epoch, batch, self.lr, loss, acc))
    with open(self.record_file, "a") as f:
      f.write("{}, epoch: {}, batch: {}, lr: {:.5}, loss: {:.3}, acc: {:.3}\n".
              format(time.strftime("%Y-%m-%d_%H:%M:%S",
                                   time.localtime()), epoch, batch, self.lr, loss, acc))


if __name__ == "__main__":
  d_pretrain = Pretrain()
  d_pretrain()
