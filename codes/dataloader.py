import json
import os
import pickle as pkl
import re
import time
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
from tqdm import tqdm

from codes.parameters import args


class DataLoader(object):
  def __init__(self, data_dir, ans_max_len, dialog_max_len=None):
    self.unkown_tok, self.pad_tok, self.sep_tok, self.bos_tok, self.eos_tok, self.time_tok, self.digit_tok = \
      ["<UNKNOWN>", "<PAD>", "<SEP>", "<BOS>", "<EOS>", "<T>", "<D>"]
    self.special_tokens = {self.unkown_tok: 1,
                           self.pad_tok: 0, self.sep_tok: 2, self.bos_tok: 3,
                           self.eos_tok: 4, self.time_tok: 5, self.digit_tok: 6}
    self.dir = data_dir
    self.ans_max_len = ans_max_len
    self.dialog_max_len = dialog_max_len

    self.is_load_dict = False
    self.data_tokens = self.preprocess(self.dir)
    self.token_id, self.id_token = self.dict_gen()
    self.data_ids, self.seq_lens = self.tokens2id()
    self.vocab_size = len(self.token_id)

  def preprocess(self, dir):
    seq_tokens = []
    for filename in tqdm(os.listdir(dir)):
      with open(os.path.join(dir, filename), "r") as file:
        episod = {"dialogue": [], "true_ans": None, "wrong_ans": []}
        file_content = file.read()
        contents = json.loads(file_content)  # 'answers', 'options', 'article', 'id'

        episod["dialogue"] = self.dialog_process(contents["article"])
        episod["true_ans"], episod["wrong_ans"] = \
          self.answers_process(list(map(lambda x: x[4:], contents["options"])), contents["answers"])
        seq_tokens.append(episod)

    for i, seq in tqdm(enumerate(seq_tokens)):
      for key in seq.keys():
        self.data_clean(seq_tokens[i][key])
    return seq_tokens

  def seq_pad(self, seq_list):
    # pad eos token in front of each sentence and bos token at the end of each sentence
    return [self.bos_tok] + seq_list + [self.eos_tok]

  def seq_clip(self, seq, max_len):
    if isinstance(seq, str):
      seq = seq.split()
    return seq[0:max_len - 2]

  def answers_process(self, answers, choice):
    assert choice in ["A", "B", "C", "D"]
    c = {"A": 0, "B": 1, "C": 2, "D": 3}
    true_ans = answers[c[choice]].split()
    if len(true_ans) > self.ans_max_len - 2:
      true_ans = true_ans[0:self.ans_max_len - 2]
    del (answers[c[choice]])
    return self.seq_pad(true_ans), \
           list(map(lambda x: self.seq_pad(self.seq_clip(x, self.ans_max_len)), answers))

  def dialog_process(self, article):
    dialogs = re.split(r"(?:f : |m : )", article.strip())
    dialogs = list(filter(None, dialogs))
    dialogs = (" " + self.sep_tok + " ").join(dialogs)
    return self.seq_pad(self.seq_clip(dialogs, self.dialog_max_len))

  def data_clean(self, tok_list):
    def is_digit(token):
      try:
        float(token)
      except ValueError as _:
        return False
      return True

    def is_valid_time(token):
      try:
        time.strptime(token, "%H:%M")
        return True
      except ValueError as _:
        return False

    for i, tok in enumerate(tok_list):
      if isinstance(tok, list):
        self.data_clean(tok)
        return

      if is_digit(tok):
        tok_list[i] = self.digit_tok
        continue
      if is_valid_time(tok):
        tok_list[i] = self.time_tok

  def dict_gen(self):
    dict_path = "./data/embedding/dict.pkl"
    if self.is_load_dict:
      try:
        with open(dict_path, "rb") as f:
          token_id = pkl.load(f)
          id_token = pkl.load(f)
          return token_id, id_token
      except FileNotFoundError:
        print("Dictionary file not found! Starting to regenerate.")

    tokens = self.preprocess("./data/mutual/train/1")
    tokens.extend(self.preprocess("./data/mutual/train/2"))
    diction = Counter()
    for data in tqdm(tokens):
      diction += Counter(data["dialogue"])
      diction += Counter(data["true_ans"])
      for item in data["wrong_ans"]:
        diction += Counter(item)
    diction = {**self.special_tokens, **dict(diction.most_common())}
    # diction = dict(filter(lambda x: x[1] > 10, diction.items()))  # set unusual tokens as <UNKNOWN>

    token_id = dict(map(lambda x, i: (x[0], i), list(diction.items()), range(len(diction))))
    id_token = {v: k for k, v in token_id.items()}

    with open(dict_path, "wb") as f:
      pkl.dump(token_id, f)
      pkl.dump(id_token, f)
    return token_id, id_token

  def tokens2id(self):
    def _tokens2id(seq):
      if isinstance(seq[0], list):  # process wrong answers
        token_ids, seq_lens = [], []
        for item in seq:
          tmp, tmp_len = _tokens2id(item)
          token_ids.append(tmp), seq_lens.append(tmp_len)
        return token_ids, seq_lens

      token_ids = list(
        map(lambda x: self.token_id[x] if x in self.token_id.keys() else self.special_tokens[self.unkown_tok], seq))
      seq_lens = len(seq)
      return token_ids, seq_lens

    def _seq_padding(seq_ids, max_len):
      if isinstance(seq_ids[0][0], list):
        _seq_ids = []
        for i, item in enumerate(seq_ids):
          _seq_ids.append(_seq_padding(item, max_len))
        return _seq_ids

      seq_ids = list(map(lambda x: x + [self.special_tokens[self.pad_tok]] * (max_len - len(x)), seq_ids))
      return seq_ids

    data_ids = {"dialogue": [], "true_ans": [], "wrong_ans": []}
    seq_lens = {"dialogue": [], "true_ans": [], "wrong_ans": []}
    max_len = {"dialogue": self.dialog_max_len, "true_ans": self.ans_max_len, "wrong_ans": self.ans_max_len}
    for seq in tqdm(self.data_tokens):
      for key in data_ids.keys():
        tmp, seq_len = _tokens2id(seq[key])
        data_ids[key].append(tmp)
        seq_lens[key].append(seq_len)

    # padding the sentences to the same length
    for key in data_ids.keys():
      if max_len[key] is None:
        max_len[key] = max(seq_lens[key])
      data_ids[key] = _seq_padding(data_ids[key], max_len[key])
    return data_ids, seq_lens

  def statistic(self):
    datas = self.data_tokens
    data_size = len(self.data_tokens)
    seq_lens = defaultdict(int)

    diction = Counter()
    for data in datas:
      seq_lens[len(data)] += 1
      diction += Counter(data)
    seq_lens = dict(sorted(seq_lens.items(), key=lambda x: x[0]))

    max_len = max(seq_lens.keys())
    prop = {}
    for i in range(max_len + 1):
      if len(prop) == 0:
        if i in seq_lens.keys():
          prop[i] = seq_lens[i] / data_size
        continue

      seq_lens.setdefault(i, 0)
      seq_lens[i] += seq_lens[i - 1]
      prop[i] = seq_lens[i] / data_size

    x = prop.keys()
    y = prop.values()
    plt.plot(x, y, '--')
    plt.show()
    pass


if __name__ == "__main__":
  dl = DataLoader(args.dev_dir, args.ans_max_len, args.dia_max_len)
