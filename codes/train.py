import json
import os
import pprint
import time

import torch as t
from torch import nn
from torch import optim

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
    self.dialog_lens, self.ans_lens, _ = self.seq_lens.values()

    self.embedding = nn.Embedding(self.vocab_size, self.embed_dim,
                                  padding_idx=self.special_tokens[self.pad_tok])
    self.generator = Generator(self.embedding, self.ans_max_len, self.g_dropout,
                               self.special_tokens, device=self.device)
    self.discriminator = Discriminator(self.embedding, device=self.device,
                                       seq_len=self.ans_max_len + self.dia_max_len)
    self.g_optim = optim.Adam(self.generator.parameters(), lr=self.g_lr0)
    self.d_optim = optim.SGD(self.discriminator.parameters(), lr=self.d_lr0)
    self = self.to(self.device)

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
    print("*" * 20 + "device: {}".format(self.device) + "*" * 20)
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
      answer, dialog, src_mask, tgt_mask, ans_lens = self.get_batch(batch)

      # caculate the state value by sampling the next action(token[t])
      # based on the current state(token[1:t-1])
      for i in range(self.max_ans_len - 1):
        states, probs = self.generator(src=dialog, tgt=answer[:, :i + 1],
                                       src_mask=src_mask, tgt_mask=tgt_mask,
                                       num_samples=self.n_samples,
                                       mode=0)
        action_values = t.zeros_like(probs)
        self.g_optim.zero_grad()

        for j in range(self.n_samples):
          state = states[j]
          scores = []

          # caculate the action value by sampling the whole sequence(token[t+1:T])
          # n times based on the state-action pair(token[1:t-1], token[t])
          for _ in range(self.n_times):
            episode = self.generator(src=dialog, tgt=state,
                                     src_mask=src_mask, tgt_mask=tgt_mask,
                                     mode=1)
            score = self.discriminator(dialog, episode).detach()
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
        generator_input = self.generator(src=dialog, tgt=generator_input,
                                         src_mask=src_mask, tgt_mask=tgt_mask,
                                         mode=2)
        print("({}: {:.3})".format(i, neg_state_value.item()), end=" ")

    #  discriminator training round
    print("\n")
    gen_sents = generator_input
    self.d_optim.zero_grad()
    loss, d_acc = self.discriminator(dialog, gen_sents.detach(), answer)
    loss.backward()
    self.d_optim.step()
    _ = self.adjust_lr(self.d_lr0, self.g_optim, self.total_batch * epoch + batch,
                       self.total_batch * self.epoch_num)

    g_acc = self.get_accuracy(gen_sents, answer)
    self.log_record(epoch, batch, loss, gen_sents, answer, g_acc, d_acc, ans_lens)
    self.model_save(epoch, batch)
    print("\n")

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

  def get_batch(self, index):
    start = index * self.batch_size
    end = min((index + 1) * self.batch_size, self.data_size)
    answer = self.answer[start:end, :]
    dialog = self.dialog[start:end, :]
    ans_lens = self.ans_lens[start:end]

    # get mask
    src_mask = (dialog == self.special_tokens[self.pad_tok]).to(dialog.device)
    tgt_mask = (answer == self.special_tokens[self.pad_tok]).to(dialog.device)
    return answer, dialog, src_mask, tgt_mask, ans_lens

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

  trainer = Trainer(mode="train")
  trainer()
  # trainer.inference("./model/model-2020-07-02_10-34-32/model-199")
