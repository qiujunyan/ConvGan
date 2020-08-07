import torch as t
from torch import nn

from codes.transformer import Transformer


class Generator(nn.Module):
  def __init__(self, embedding, ans_max_len, special_tokens,
               hidden_dim=64, num_layers=1, num_heads=8,
               n_times=20, is_cuda=True, mode="train"):
    super(Generator, self).__init__()
    self.embedding = embedding
    self.ans_max_len = ans_max_len
    self.vocab_size, self.embed_dim = self.embedding.weight.shape
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.mode = mode
    self.pad_id = special_tokens["<PAD>"]
    self.bos_id = special_tokens["<BOS>"]
    self.is_cuda = is_cuda
    self.curriculum_rate = 0  # used to control the probability of replacing the true tokens with generated ones.
    self.n_times = n_times

    self.encoder = Transformer(self.embed_dim, self.vocab_size,
                               self.hidden_dim, self.num_heads, self.num_layers)
    self.decoder = Transformer(self.embed_dim, self.vocab_size,
                               self.hidden_dim, self.num_heads, self.num_layers)
    if self.is_cuda:
      self = self.cuda()

  def forward(self, dialogs, mode=None, init_dec_input_index=None,
              num_samples=10, is_pretraining=False):
    self.dialogs = self.embedding(dialogs)
    enc_output = self.encoder(self.dialogs, ori_k=self.dialogs)
    batch_size = dialogs.size(0)
    init_dec_input_index = self.get_init_dec_input_index(init_dec_input_index, batch_size)
    dec_seq_len = init_dec_input_index.size(1)

    # pretraining
    if is_pretraining:
      return self.gen_whole_sequence(enc_output, init_dec_input_index)

    if mode == 0:  # sample next tokens
      next_token_index, probs = self.sample_next_token(init_dec_input_index, enc_output, num_samples)
      init_dec_input_index = init_dec_input_index.repeat([1, num_samples]).split(dec_seq_len, -1)
      next_token_index = next_token_index.split(1, -1)
      return list(map(lambda x, y: t.cat([x, y], -1), init_dec_input_index, next_token_index)), probs.transpose(1, 0)

    elif mode == 1:  # apply rollout algorithm to generate the whole sequence
      return self.sample_whole_sequence(init_dec_input_index, enc_output)

    elif mode == 2:  # generate the next token
      next_token_index = self.predict_next_token(init_dec_input_index, enc_output)
      return t.cat([init_dec_input_index, next_token_index], -1)

  def get_init_dec_input_index(self, init_dec_input_index, batch_size):
    if init_dec_input_index is None:
      init_dec_input_index = t.ones(batch_size, 1, dtype=t.int64) * self.bos_id
    if self.is_cuda:
      init_dec_input_index = init_dec_input_index.cuda()
    return init_dec_input_index

  def gen_whole_sequence(self, enc_output, init_dec_input_index, curriculum_rate=0):
    batch_size, dec_seq_len = init_dec_input_index.size()
    outputs = t.zeros(batch_size, self.ans_max_len - 1, self.vocab_size)
    if self.is_cuda:
      outputs = outputs.cuda()

    for i in range(self.ans_max_len - 1):
      outputs[:, i, :] = self.decoder(self.embedding(init_dec_input_index[:, :i + 1]),
                                      enc_output, enc_output,
                                      ori_k=self.dialogs, mode="decode")  # B * V
      # schedual sampling
      # if curriculum_rate > t.rand(1):
      #   init_dec_input_index[:, i + 1] = t.multinomial(outputs[:, i, :], 1)
    return outputs

  def sample_next_token(self, init_dec_input_index, enc_output, num_samples=10):
    def _get_probs(dec_output, index):
      assert dec_output.size(0) == index.size(0)
      probs = t.zeros_like(index, dtype=t.float32)
      batch_size = dec_output.size(0)
      for i in range(batch_size):
        probs[i, :] = dec_output[i, index[i]]
      return probs

    small_number = t.Tensor([1e-8]).cuda() if self.is_cuda else t.Tensor([1e-8])
    dec_input = self.embedding(init_dec_input_index)
    dec_output = self.decoder(dec_input, enc_output, enc_output,
                              ori_k=self.dialogs, mode="decode").squeeze(1)
    num_samples = min(num_samples, self.vocab_size)
    next_token_index = t.multinomial(dec_output + small_number, num_samples)
    return next_token_index, _get_probs(dec_output, next_token_index)
    # return next_token_index, dec_output

  def predict_next_token(self, init_dec_input_index, enc_output):
    dec_input = self.embedding(init_dec_input_index)
    dec_output = self.decoder(dec_input, enc_output, enc_output,
                              ori_k=self.dialogs, mode="decode")
    next_token_index = t.argmax(dec_output, -1).unsqueeze(-1)
    return next_token_index

  def sample_whole_sequence(self, init_dec_input_index, enc_output):
    batch_size, dec_seq_len = init_dec_input_index.size()
    pad_index = t.ones(batch_size, self.ans_max_len - dec_seq_len, dtype=t.int64) * self.pad_id
    if self.is_cuda: pad_index = pad_index.cuda()
    dec_input_index = t.cat([init_dec_input_index, pad_index], 1)

    for i in range(dec_seq_len, self.ans_max_len):
      dec_input = self.embedding(dec_input_index[:, :i])
      dec_output = self.decoder(dec_input, enc_output, enc_output,
                                ori_k=self.dialogs, mode="decode")  # B * 1 * V
      dec_input_index[:, i:i + 1] = t.multinomial(dec_output, 1)
    return dec_input_index


if __name__ == "__main__":
  pass
