import torch as t
from torch import nn

from codes.transformer import Encoder, Decoder


class Generator(nn.Module):
  def __init__(self, embedding, ans_max_len, dropout, special_tokens,
               hidden_dim=64, d_ff=128, num_layers=1, num_heads=8,
               n_times=20, device="cuda:0", mode="train"):
    super(Generator, self).__init__()
    self.embedding = embedding
    self.ans_max_len = ans_max_len
    self.vocab_size, self.embed_dim = self.embedding.weight.shape
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.mode = mode
    self.pad_id = special_tokens["<PAD>"]
    self.bos_id = special_tokens["<BOS>"]
    self.device = device
    self.dropout = dropout
    self.curriculum_rate = 0  # used to control the probability of replacing the true tokens with generated ones.
    self.n_times = n_times

    self.encoder = Encoder(hidden_dim=self.embed_dim, num_layers=1,
                           d_ff=d_ff, num_heads=num_heads)
    self.decoder = Decoder(hidden_dim=self.embed_dim, vocab_size=self.vocab_size, num_layers=1,
                           d_ff=d_ff, num_heads=num_heads)
    self = self.to(device)

  def forward(self, src, tgt=None, src_mask=None, tgt_mask=None,
              num_samples=10, is_pretraining=False, mode=None):
    self.src_mask, self.tgt_mask = src_mask, tgt_mask
    self.src_embed = self.embedding(src)
    enc_output = self.encoder(self.src_embed, src_mask)
    batch_size = src.size(0)
    tgt = self.get_init_dec_input_index(tgt, batch_size)
    dec_seq_len = tgt.size(1)

    # pretraining
    if is_pretraining:
      return self.gen_whole_sequence(enc_output, tgt)

    if mode == 0:  # sample next tokens
      next_token_index, probs = self.sample_next_token(tgt, enc_output, num_samples)
      tgt = tgt.repeat([1, num_samples]).split(dec_seq_len, -1)
      next_token_index = next_token_index.split(1, -1)
      return list(map(lambda x, y: t.cat([x, y], -1), tgt, next_token_index)), probs.transpose(1, 0)

    elif mode == 1:  # apply rollout algorithm to generate the whole sequence
      return self.sample_whole_sequence(tgt, enc_output)

    elif mode == 2:  # generate the next token
      next_token_index = self.predict_next_token(tgt, enc_output)
      return t.cat([tgt, next_token_index], -1)

  def get_init_dec_input_index(self, tgt, batch_size):
    if tgt is None:
      tgt = t.ones(batch_size, 1, dtype=t.int64) * self.bos_id
    tgt = tgt.to(self.device)
    return tgt

  def gen_whole_sequence(self, enc_output, tgt, curriculum_rate=0):
    batch_size, dec_seq_len = tgt.size()
    outputs = self.decoder(self.embedding(tgt[:, :-1]),
                           enc_output, enc_output,
                           q_mask=self.tgt_mask[:, :-1], k_mask=self.src_mask)  # B * l-1 * V
    return outputs

  def sample_next_token(self, tgt, enc_output, num_samples=10):
    def _get_probs(dec_output, index):
      assert dec_output.size(0) == index.size(0)
      probs = t.zeros_like(index, dtype=t.float32)
      batch_size = dec_output.size(0)
      for i in range(batch_size):
        probs[i, :] = dec_output[i, index[i]]
      return probs

    small_number = t.Tensor([1e-8], device=enc_output.device)
    dec_input = self.embedding(tgt)
    dec_output = self.decoder(dec_input, enc_output, enc_output,
                              q_mask=self.src_mask, k_mask=self.tgt_mask).squeeze(1)
    num_samples = min(num_samples, self.vocab_size)
    next_token_index = t.multinomial(dec_output + small_number, num_samples)
    return next_token_index, _get_probs(dec_output, next_token_index)
    # return next_token_index, dec_output

  def predict_next_token(self, tgt, enc_output):
    dec_input = self.embedding(tgt)
    dec_output = self.decoder(dec_input, enc_output, enc_output,
                              q_mask=self.src_mask, k_mask=self.tgt_mask)
    next_token_index = t.argmax(dec_output, -1).unsqueeze(-1)
    return next_token_index

  def sample_whole_sequence(self, tgt, enc_output):
    batch_size, dec_seq_len = tgt.size()
    pad_index = t.ones(batch_size, self.ans_max_len - dec_seq_len,
                       dtype=t.int64, device=enc_output.device) * self.pad_id
    dec_input_index = t.cat([tgt, pad_index], 1)

    for i in range(dec_seq_len, self.ans_max_len):
      dec_input = self.embedding(dec_input_index[:, :i])
      dec_output = self.decoder(dec_input, enc_output, enc_output,
                                q_mask=self.src_mask, k_mask=self.tgt_mask)  # B * 1 * V
      dec_input_index[:, i:i + 1] = t.multinomial(dec_output, 1)
    return dec_input_index


if __name__ == "__main__":
  pass
