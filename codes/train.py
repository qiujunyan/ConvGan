import json
import os
import pprint
import time

import torch as t
from torch import nn
from torch import optim
import math
from codes.dataloader import DataLoader
from codes.discriminator import Discriminator
from codes.generator import Generator
from codes.parameters import Args


class Trainer(nn.Module, DataLoader, Args):
  def __init__(self, data_dir="data/mutual/train"):
    nn.Module.__init__(self)
    Args.__init__(self)
    DataLoader.__init__(self, data_dir, is_load_dict=True)
    self.dialog = t.LongTensor(self.data_ids["dialogue"]).to(self.device)
    self.answer = t.LongTensor(self.data_ids["true_ans"]).to(self.device)
    self.dialog_lens = t.LongTensor(self.seq_lens["dialogue"]).to(self.device)
    self.ans_lens = t.LongTensor(self.seq_lens["true_ans"]).to(self.device)
    self.total_batch = math.ceil(self.data_size / self.batch_size)
    # self.total_batch = 2

    self.embedding = nn.Embedding(self.vocab_size, self.embed_dim,
                                  padding_idx=self.special_tokens[self.pad_tok])
    self.generator = Generator(self.embedding, self.ans_max_len, self.g_dropout,
                               self.special_tokens, self.d_ff, self.g_num_layers,
                               self.num_heads, self.n_times, device=self.device)
    self.discriminator = Discriminator(self.embedding,
                                       seq_len=self.ans_max_len + self.dia_max_len,
                                       device=self.device)
    self.g_optim = optim.Adam(self.generator.parameters(), lr=self.g_lr0)
    self.d_optim = optim.Adam(self.discriminator.parameters(), lr=self.d_lr0)
    # self.init_params()
    self = self.to(self.device)

  def forward(self):
    self.load_generator("./pretrain/generator/model/model-2020-08-14_19-07-52/model-200-0")
    self.load_discriminator("./pretrain/discriminator/model/model-2020-08-14_13-48-29/model-99-27")
    self.model_dir = os.path.join("model", "model-{}".
                                  format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    if not os.path.isdir(self.model_dir):
      os.mkdir(self.model_dir)
    self.record_file = os.path.join(self.model_dir, "log.txt")
    self.print_train_info()
    for epoch in range(self.epoch_num):
      self.train_one_epoch(epoch)

  def print_train_info(self):
    print("*" * 20 + "device: {}, total_batch: {}".
          format(self.device, self.total_batch) + "*" * 20)
    self.record_params()

  def init_params(self):
    for param_name, param_value in self.discriminator.named_parameters():
      if param_value.dim() < 2:
        continue
      if param_name == "embedding.weight":
        nn.init.xavier_normal_(param_value[1:, :])
        continue
      nn.init.xavier_normal_(param_value)

  def train_one_epoch(self, epoch):
    for batch in range(self.total_batch):
      self.lr = self.adjust_lr(self.g_lr0, self.g_optim,
                               epoch * self.total_batch + batch,
                               self.epoch_num * self.total_batch,
                               min_lr=1e-4, warmup=10, warmup_lr=1e-3)
      src, tgt, src_mask, tgt_mask, ans_lens = self.get_batch(batch)

      # caculate the state value by sampling the next action(token[t])
      # based on the current state(token[1:t-1])
      for i in range(self.ans_max_len - 1):
        states, probs = self.generator(src=src, tgt=tgt[:, :i + 1],
                                       src_mask=src_mask, tgt_mask=tgt_mask,
                                       num_samples=self.n_samples, mode=1)
        action_values = t.zeros_like(probs)
        self.g_optim.zero_grad()

        for j in range(self.n_samples):
          state = states[j]
          scores = []

          # caculate the action value by sampling the whole sequence(token[t+1:T])
          # n times based on the state-action pair(token[1:t-1], token[t])
          for _ in range(self.n_times):
            episode = self.generator(src=src, tgt=state,
                                     src_mask=src_mask, tgt_mask=tgt_mask,
                                     mode=2)
            score = self.discriminator(src, episode).detach()
            scores.append(score)
          action_value = sum(scores) / len(scores)
          action_values[j, :] = action_value

        #  generator training round
        neg_state_value = -t.mean(action_values * probs)
        neg_state_value.backward()
        # self.clip_grad_value(self.generator.decoder.output_linear.weight, 1e-33)
        self.g_optim.step()
        print("({}: {:.3})".format(i, neg_state_value.item()), end=" ")

      gen_sents = self.generator(src=src, src_mask=src_mask, mode=3)

    #  discriminator training round
    print("\n")
    self.d_optim.zero_grad()
    loss, d_acc = self.discriminator(src, gen_sents.detach(), tgt)
    loss.backward()
    self.d_optim.step()
    _ = self.adjust_lr(self.d_lr0, self.g_optim, self.total_batch * epoch + batch,
                       self.total_batch * self.epoch_num)

    g_acc = self.get_accuracy(gen_sents, tgt)
    self.log_record(epoch, batch, loss, gen_sents, tgt, g_acc, d_acc, ans_lens)
    self.model_save(epoch, batch)
    print("\n")

  def adjust_lr(self, lr0, optim, iter, max_iter,
                min_lr=None, warmup=None,
                warmup_lr=1e-3, power=2):
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

  def get_batch(self, index):
    start = index * self.batch_size
    end = min((index + 1) * self.batch_size, self.data_size)
    tgt = self.answer[start:end, :]
    src = self.dialog[start:end, :]
    ans_lens = self.ans_lens[start:end]

    # get mask
    src_mask = (src == self.special_tokens[self.pad_tok]).to(src.device)
    tgt_mask = (tgt == self.special_tokens[self.pad_tok]).to(src.device)
    return src, tgt, src_mask, tgt_mask, ans_lens

  def shuffle_data(self):
    shuffle_index = t.randperm(self.answer.size(0))
    self.answer = self.answer[shuffle_index]
    self.dialog = self.dialog[shuffle_index]
    self.ans_lens = self.ans_lens[shuffle_index]

  def record_params(self):
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

  def log_record(self, epoch, batch, loss,
                 gen_datas, true_datas,
                 g_acc=0.0, d_acc=0.0, ans_lens=None):
    seq = self.sent_gen(gen_datas)
    real_seq = self.sent_gen(true_datas, ans_lens)
    print("epoch {}, batch: {}, lr: {:.5}, loss: {:.3}, g_acc: {:.3}, d_acc: {:.3}".
          format(epoch, batch, self.lr, loss, g_acc, d_acc))
    print("g: {}\nr: {}".format(seq, real_seq))

    # if self.mode in ["train", "t", "T"]:
    self.record_train_process(epoch, batch, loss, g_acc, d_acc, real_seq, seq)

  def record_train_process(self, epoch, batch, loss, g_acc, d_acc, r, g):
    with open(self.record_file, "a", encoding="utf-8") as f:
      f.write("{}, epoch: {}, batch: {}, lr: {:.5} loss: {:.3}, gacc: {:.3}, dacc: {:.3}\n".
              format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
                     epoch, batch, self.lr, loss, g_acc, d_acc))
      try:
        f.write("r: {}\ng: {}\n".format(r, g))
      except UnicodeDecodeError:
        print("Encode error occured when writing sentence: \n{}!".format(g))

  def sent_gen(self, batch_seq, seq_len=None):
    seq = batch_seq[0] if seq_len is None else batch_seq[0][:seq_len[0]]
    return " ".join(list(map(lambda id: self.id_token[id], seq.tolist())))

  def get_accuracy(self, gen_data, true_data):
    is_target = (true_data != self.special_tokens[self.pad_tok])
    hit_num = (gen_data == true_data) & is_target
    return t.sum(hit_num.float()) / t.sum(is_target.float())

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

  trainer = Trainer()
  trainer()
  # trainer.inference("./model/model-2020-07-02_10-34-32/model-199")
