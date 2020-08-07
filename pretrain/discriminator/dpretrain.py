import os
import time

import numpy as np
import torch as t

from codes.parameters import args
from codes.train import Trainer


class Pretrain(Trainer):
  def __init__(self, args):
    super(Pretrain, self).__init__(args, mode="p")
    self.d_lr0 = 0.1
    self.neg_datas = np.array(self.data_loader.data_ids["wrong_ans"])
    del (self.generator)

  def forward(self):
    self.model_dir = os.path.join("./pretrain/discriminator/model", "model-{}".
                                  format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    self.load_discriminator("./pretrain/discriminator/model/model-2020-08-05_09-58-27/model-160-41")
    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    self.record_file = os.path.join(self.model_dir, "record.txt")
    self.print_train_info()
    for epoch in range(self.epoch_num):
      self.train_one_epoch(epoch)

  def train_one_epoch(self, epoch):
    self.lr = self.adjust_lr(self.d_lr0, epoch, self.epoch_num,
                             warmup=10, warmup_lr=1e-2,
                             power=10, min_lr=1e-7)
    for batch in range(self.total_batch):
      self.d_optim.zero_grad()
      answer, neg_answer, dialog = self.batch_data_prep(batch)
      loss, acc = self.discriminator(dialog, neg_answer, answer)

      loss.backward()
      self.d_optim.step()
    self.log_record(epoch, batch, loss, acc)
    self.model_save(epoch, batch)

  def batch_data_prep(self, index):
    start = index * self.bsize
    end = min((index + 1) * self.bsize, self.data_size)
    answer = self.data_converter(self.ans_datas[start:end, :])
    neg_answer = self.data_converter(self.neg_datas[start:end, :])
    dialog = self.data_converter(self.dialog_datas[start:end, :])
    return answer, neg_answer, dialog

  def adjust_lr(self, iter, max_iter, min_lr=None, warmup=None, warmup_lr=1e-3, power=5):
    if warmup is not None and iter < warmup:
      lr = warmup_lr
    else:
      lr = self.lr0 * ((1 - float(iter) / max_iter) ** power)
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
  d_pretrain = Pretrain(args)
  d_pretrain()
