import torch as t
from torch import nn
import math
from codes.transformer import Encoder, Decoder


class Generator(nn.Module):
  def __init__(self, embedding, ans_max_len, dropout, special_tokens,
               d_ff=512, num_layers=6, num_heads=8,
               n_times=10, device="cuda:0", mode="train"):
    super(Generator, self).__init__()
    self.embedding = embedding
    self.ans_max_len = ans_max_len
    self.vocab_size, self.embed_dim = self.embedding.weight.shape
    self.mode = mode
    self.pad_id = special_tokens["<PAD>"]
    self.bos_id = special_tokens["<BOS>"]
    self.device = device
    self.curriculum_rate = 0  # used to control the probability of replacing the true tokens with generated ones.
    self.n_times = n_times

    self.encoder = Encoder(hidden_dim=self.embed_dim, d_ff=d_ff, num_heads=num_heads,
                           num_layers=num_layers, dropout=dropout)
    self.decoder = Decoder(hidden_dim=self.embed_dim, vocab_size=self.vocab_size,
                           d_ff=d_ff, num_heads=num_heads,
                           num_layers=num_layers, dropout=dropout)
    self = self.to(device)

  def forward(self, src, tgt=None, src_mask=None, tgt_mask=None,
              num_samples=10, is_pretraining=False, mode=None):
    self.src_mask, self.tgt_mask = src_mask, tgt_mask
    enc_output = self.encoder(self.get_embedding(src), src_mask)
    batch_size = src.size(0)
    tgt = self.get_init_dec_input_index(tgt, batch_size)

    switch = [self.gen_whole_sequence, self.sample_next_token,
              self.sample_whole_sequence, self.predict_next_token]
    return switch[mode](tgt, enc_output, num_samples)

  def get_init_dec_input_index(self, tgt, batch_size):
    if tgt is None:
      tgt = t.ones(batch_size, 1, dtype=t.int64) * self.bos_id
    return tgt.to(self.device)

  def get_embedding(self, index, scale=True):
    embedding = self.embedding(index)
    if scale:
      embedding = embedding * math.sqrt(self.embed_dim)
    return embedding

  def gen_whole_sequence(self, tgt, enc_output, num_samples=None):
    batch_size, dec_seq_len = tgt.size()
    outputs = self.decoder(self.get_embedding(tgt[:, :-1]),
                           enc_output, enc_output,
                           q_mask=self.src_mask, k_mask=self.tgt_mask)  # B * l-1 * V
    return outputs

  def sample_next_token(self, tgt, enc_output, num_samples=10):
    # TODO
    def _get_probs(dec_output, index):
      batch_size = dec_output.size(0)
      batch_index = t.arange(batch_size).view(batch_size, -1)
      return dec_output[batch_index, index]

    small_number = t.Tensor([1e-8]).to(device=self.device)
    dec_input = self.get_embedding(tgt)
    tgt_mask = (tgt == self.pad_id)
    dec_output = self.decoder(dec_input, enc_output, enc_output,
                              q_mask=tgt_mask, k_mask=self.src_mask)
    dec_output = dec_output[:, -1, :]  # last output token is needed
    num_samples = min(num_samples, self.vocab_size)
    next_token_index = t.multinomial(dec_output + small_number, num_samples)

    dec_seq_len = tgt.size(1)
    tgt = tgt.repeat([1, num_samples]).split(dec_seq_len, -1)
    probs = _get_probs(dec_output, next_token_index)
    next_token_index = next_token_index.split(1, -1)
    return list(map(lambda x, y: t.cat([x, y], -1), tgt, next_token_index)), \
           probs.transpose(1, 0)

  def predict_next_token(self, tgt, enc_output, num_samples=None):
    dec_input = self.get_embedding(tgt)
    tgt_mask = (tgt == self.pad_id)
    dec_output = self.decoder(dec_input, enc_output, enc_output,
                              q_mask=tgt_mask, k_mask=self.src_mask)
    return t.argmax(dec_output, -1)

  def sample_whole_sequence(self, tgt, enc_output, num_samples=None):
    batch_size = enc_output.size(0)
    bos_index = t.ones([batch_size, 1], dtype=t.int64, device=self.device) * self.bos_id
    dec_input_index = bos_index

    for i in range(1, self.ans_max_len):
      dec_input = self.get_embedding(dec_input_index)
      tgt_mask = (dec_input_index == self.pad_id).to(device=self.device)
      dec_output = self.decoder(dec_input, enc_output, enc_output,
                                q_mask=tgt_mask, k_mask=self.src_mask)  # B * 1 * V
      dec_input_index = t.cat([bos_index, t.argmax(dec_output, -1)], -1)
    return dec_input_index


if __name__ == "__main__":
  pass
