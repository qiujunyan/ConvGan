import json
import math
import os
import pprint
import time

import numpy as np
import torch as t
from torch import nn
from torch import optim

from codes.dataloader import DataLoader
from codes.discriminator import Discriminator
from codes.generator import Generator
from codes.parameters import args


class Trainer(nn.Module):
  def __init__(self, args, mode="t"):
    super(Trainer, self).__init__()
    self.mode = mode
    self.is_cuda = args.is_cuda
    self.data_loader = DataLoader(self.mode_check(mode, args), args.ans_max_len, args.dia_max_len)
    self.dialog_datas = np.array(self.data_loader.data_ids["dialogue"])
    self.ans_datas = np.array(self.data_loader.data_ids["true_ans"])
    self.max_dialog_len = self.dialog_datas.shape[1]
    self.max_ans_len = self.ans_datas.shape[1]
    self.dialog_lens, self.ans_lens, _ = self.data_loader.seq_lens.values()
    self.data_tokens = self.data_loader.data_tokens
    self.data_size, _ = self.ans_datas.shape
    self.pad_tok = self.data_loader.pad_tok

    # dataloader parameters
    self.vocab_size = self.data_loader.vocab_size
    self.token_id = self.data_loader.token_id
    self.id_token = self.data_loader.id_token
    self.special_tokens = self.data_loader.special_tokens

    # parameters related to training
    self.args = args
    self.embed_dim = args.embed_dim
    self.hidden_dim = args.hidden_dim
    self.bsize = args.batch_size
    self.epoch_num = args.epoch_num
    self.g_lr0 = args.g_lr
    self.d_lr0 = args.d_lr
    self.g_dropout = args.dropout
    self.channels = args.channels
    self.decay_rate = args.decay_rate
    self.n_times = args.n_times
    self.device = args.device
    self.clip = args.clip
    self.num_samples = args.num_samples
    self.curriculum_rate = args.curriculum_rate
    self.total_batch = math.ceil(self.data_size / self.bsize)

    self.embedding = nn.Embedding(self.vocab_size, self.embed_dim,
                                  padding_idx=self.special_tokens[self.pad_tok])
    self.generator = Generator(self.embedding, self.max_ans_len, self.g_dropout,
                               self.special_tokens, is_cuda=self.is_cuda)
    self.discriminator = Discriminator(self.embedding,
                                       seq_len=self.max_ans_len + self.max_dialog_len,
                                       is_cuda=self.is_cuda)
    self.g_optim, self.d_optim = self.init_optim()

  def mode_check(self, mode, args):
    if mode.lower() in ["train", "t"]:
      return args.train_dir1
    if mode.lower() in ["pretrain", "p"]:
      return args.train_dir2
    elif mode.lower() in ["eval", "e"]:
      return args.dev_dir
    else:
      raise ValueError

  def forward(self):
    # self.load_generator("./pretrain/generator/model/model-2020-08-02_19-14-25/model-618-40")
    self.load_discriminator("./model/model-2020-08-07_00-30-31/model-10-0")
    self.model_dir = os.path.join(self.args.model_dir, "model-{}".
                                  format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    self.record_file = os.path.join(self.model_dir, "record.txt")
    self.print_train_info()
    for epoch in range(self.epoch_num):
      self.train_one_epoch(epoch)

  def print_train_info(self):
    print("*" * 20 + "device: {}".format("gpu" if self.is_cuda else "cpu") + "*" * 20)
    self.record_params()

  def train_one_epoch(self, epoch):
    with open(self.record_file, "a") as f:
      f.write("\n")
    self.total_batch = 2
    generator_input = None
    for batch in range(1, self.total_batch):
      self.lr = self.adjust_lr(self.g_lr0, self.g_optim,
                               epoch * self.total_batch + batch,
                               self.epoch_num * self.total_batch,
                               min_lr=1e-4, warmup=10, warmup_lr=1e-3)
      ans_datas, dialog_datas, ans_lens = self.batch_data_prep(batch)

      # caculate the state value by sampling the next action(token[t])
      # based on the current state(token[1:t-1])
      for i in range(self.max_ans_len - 1):
        states, probs = self.generator(dialogs=dialog_datas,
                                       mode=0,
                                       init_dec_input_index=ans_datas[:, :i + 1],
                                       num_samples=self.num_samples)
        probs.retain_grad()
        action_values = t.zeros_like(probs)
        self.g_optim.zero_grad()

        for j in range(self.num_samples):
          state = states[j]
          scores = []

          # caculate the action value by sampling the whole sequence(token[t+1:T])
          # n times based on the state-action pair(token[1:t-1], token[t])
          for _ in range(self.n_times):
            episode = self.generator(dialogs=dialog_datas,
                                     mode=1,
                                     init_dec_input_index=state)
            score = self.discriminator(dialog_datas, episode).detach()
            scores.append(score)
          action_value = sum(scores) / len(scores)
          action_values[j, :] = action_value

        #  generator training round
        neg_state_value = -t.mean(action_values * probs)
        neg_state_value.backward()
        # self.clip_grad_value(self.generator.decoder.output_linear.weight, 1e-33)
        self.g_optim.step()
        # generated token as the next token, this may cause accumulative error
        # as the training process goes.
        generator_input = self.generator(dialogs=dialog_datas,
                                         mode=2,
                                         init_dec_input_index=generator_input)
        print("({}: {:.3})".format(i, neg_state_value.item()), end=" ")

    #  discriminator training round
    print("\n")
    gen_sents = generator_input
    self.d_optim.zero_grad()
    loss, d_acc = self.discriminator(dialog_datas, gen_sents.detach(), ans_datas)
    loss.backward()
    self.d_optim.step()
    _ = self.adjust_lr(self.d_lr0, self.g_optim, self.total_batch * epoch + batch,
                       self.total_batch * self.epoch_num)

    g_acc = self.get_accuracy(gen_sents, ans_datas)
    self.log_record(epoch, batch, loss, gen_sents, ans_datas, g_acc, d_acc, ans_lens)
    self.model_save(epoch, batch)
    print("\n")

  def init_optim(self):
    g_optim = optim.Adam(self.generator.parameters(), lr=self.g_lr0)
    d_optim = optim.SGD(self.discriminator.parameters(), lr=self.d_lr0)
    return g_optim, d_optim

  def adjust_curriculum_rate(self, iter, max_iter):
    iter = min(max_iter, iter)
    return iter / max_iter

  def adjust_lr(self, lr0, optim, iter, max_iter,
                min_lr=None, warmup=None,
                warmup_lr=1e-3, power=10):
    if warmup is not None and iter < warmup:
      return warmup_lr
    lr = lr0 * ((1 - float(iter) / max_iter) ** power)
    if min_lr is not None:
      lr = max(min_lr, lr)
    optim.param_groups[0]["lr"] = lr
    return lr

  def clip_grad_norm(self, parameters, clip_norm):
    t.nn.utils.clip_grad_norm_(parameters, clip_norm)

  def clip_grad_value(self, parameters, clip_value):
    t.nn.utils.clip_grad_value_(parameters, clip_value)

  def data_converter(self, data):
    ''' from lists with random length to LongTensor'''
    return t.LongTensor(data).cuda() if self.is_cuda else t.LongTensor(data)

  def record_params(self):
    # record_info = {"batch_num": self.total_batch,
    #                "vocab size": self.vocab_size,
    #                "embedding dim": self.embed_dim,
    #                "batch size": self.bsize,
    #                "total epoch number": self.epoch_num,
    #                "generator learning rate": self.g_lr0,
    #                "discriminator learning rate": self.d_lr0,
    #                "answer_max_len": self.max_ans_len,
    #                "dialog_max_len": self.max_dialog_len,
    #                "next_token_samples": self.num_samples,
    #                "n_times": self.n_times,
    #                "use_cuda": True if self.is_cuda else False}
    record_info = {}
    base_type = (str, int, float, bool)
    for attr_name in dir(self):
      if attr_name.startswith("_"):
        continue
      attr_value = getattr(self, attr_name)
      if isinstance(attr_value, base_type):
        record_info[attr_name] = attr_value

    pprint.pprint(record_info)
    with open(self.record_file, "w") as f:
      json.dump(record_info, f, separators=("\n", " : "))

  def batch_data_prep(self, index):
    start = index * self.bsize
    end = min((index + 1) * self.bsize, self.data_size)
    ans_datas = self.data_converter(self.ans_datas[start:end, :])
    dialog_datas = self.data_converter(self.dialog_datas[start:end, :])
    ans_lens = self.ans_lens[start:end]
    return ans_datas, dialog_datas, ans_lens

  def log_record(self, epoch, batch, loss,
                 gen_datas, true_datas,
                 g_acc=0.0, d_acc=0.0, ans_lens=None):
    seq = self.sent_gen(gen_datas)
    real_seq = self.sent_gen(true_datas, ans_lens)
    print("g: {}\nr: {}".format(seq, real_seq))
    print("epoch {}, batch: {}, lr: {:.5}, loss: {:.3}, g_acc: {:.3}, d_acc: {:.3}".
          format(epoch, batch, self.lr, loss, g_acc, d_acc))

    # if self.mode in ["train", "t", "T"]:
    self.record_train_process(epoch, batch, loss, g_acc, d_acc, real_seq, seq)

  def record_train_process(self, epoch, batch, loss, g_acc, d_acc, r, g):
    with open(self.record_file, "a", encoding="utf-8") as f:
      f.write("{}, epoch: {}, batch: {}, lr: {:.5} loss: {:.3}, gacc: {:.3}, dacc: {:.3}\n".
              format(time.strftime("%Y-%m-%d_%H:%M:%S",
                                   time.localtime()), epoch, batch, self.lr, loss, g_acc, d_acc))
      try:
        f.write("r: {}\ng: {}\n".format(r, g))
      except UnicodeDecodeError:
        print("Encode error occured when writing sentence: \n{}!".format(g))

  def sent_gen(self, batch_seq, seq_len=None):
    seq = batch_seq[0] if seq_len is None else batch_seq[0][:seq_len[0]]
    return " ".join(list(map(lambda id: self.id_token[id], seq.tolist())))

  def get_accuracy(self, gen_data, true_data):
    return t.mean((gen_data == true_data).float())

  def load_generator(self, model_path):
    try:
      self.generator.load_state_dict(t.load(model_path)["generator_state_dict"])
      print("Generator loaded!")
    except FileNotFoundError:
      print("Generator not found!")
    except RuntimeError:
      print("Generator State dict mismatched!")

  def load_discriminator(self, model_path):
    try:
      self.discriminator.load_state_dict(t.load(model_path)["discriminator_state_dict"])
      print("Discriminator loaded!")
    except FileNotFoundError:
      print("Discriminator not found!")
    except RuntimeError:
      print("Discriminator State dict mismatched!")

  def model_save(self, epoch, batch):
    save_path = os.path.join(self.model_dir,
                             "model-{}-{}".format(epoch, batch))
    t.save({"epoch": epoch,
            "batch": batch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "lr": self.lr}, save_path)


if __name__ == "__main__":
  print("data loaded...")

  trainer = Trainer(args, mode="train")
  trainer()
  # trainer.inference("./model/model-2020-07-02_10-34-32/model-199")
