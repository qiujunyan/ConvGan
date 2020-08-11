class Args(object):
  def __init__(self):
    self.batch_size = 60
    self.epoch_num = 1000
    self.embed_dim = 64
    self.d_ff = 128
    self.ans_max_len = 20
    self.dia_max_len = 400

    self.pad_tok = "<PAD>"
    self.unkown_tok = "<UNKNOWN>"
    self.sep_tok = "<SEP>"
    self.bos_tok = "<BOS>"
    self.eos_tok = "<EOS>"
    self.time_tok = "<T>"
    self.digit_tok = "<D>"
    self.special_tokens = {self.pad_tok: 0, self.unkown_tok: 1,
                           self.sep_tok: 2, self.bos_tok: 3,
                           self.eos_tok: 4, self.time_tok: 5, self.digit_tok: 6}
    self.clean_data = False
    self.channels = [16, 32]
    self.g_lr0 = 1e-3
    self.d_lr0 = 1e-3
    self.g_dropout = 0.0
    self.d_dropout = 0.0
    self.device = "cuda:0"
    self.n_samples = 5
    self.n_times = 10
    self.reg_rate = 1e-3
    self.train_dir = "data/mutual/train/"
    self.dev_dir = "data/mutual/dev/"
    self.test_dir = "data/mutual/test/"
    self.model_dir = "model"
