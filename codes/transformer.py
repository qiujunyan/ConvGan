import torch as t
from torch import nn


class Transformer(nn.Module):
  def __init__(self, embed_dim, vocab_size, hidden_dim=64,
               num_heads=8, num_layers=6, dropout=0):
    super(Transformer, self).__init__()
    self.num_layers = num_layers

    self.pe = PositionalEncoding(embed_dim)
    self.input_linear = nn.Linear(embed_dim, hidden_dim)
    self.multi_head_attention = nn.ModuleList([MultiHeadAttention(embed_dim, hidden_dim, num_heads)] * num_layers)
    self.layer_norm = LayerNormalization(hidden_dim)
    self.feed_forward = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)] * num_layers)
    self.activation_function = t.relu
    self.output_linear = nn.Linear(hidden_dim, vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, key=None, value=None, ori_k=None, mode="e"):
    '''post layer normalization'''
    query = self.pe(query)
    if mode.lower() in ["e", "encoder", "encode"]:
      for i in range(self.num_layers):
        # sublayer 1
        query = self.layer_norm(query)
        outputs = self.dropout(query + self.multi_head_attention[i](query, query, query, ori_k))
        # sublayer 2
        outputs = self.dropout(outputs + self.activation_function(self.feed_forward[i](self.layer_norm(outputs))))
        query = outputs
      return query
    elif mode.lower() in ["d", "decoder", "decode"]:
      for i in range(self.num_layers):
        # sublayer 1
        query = self.layer_norm(query)
        query = self.dropout(query + self.multi_head_attention[i](query, query, query))
        # sublsyer 2
        outputs = self.dropout(
          query + self.multi_head_attention[i](self.layer_norm(query), key, value, ori_k))  # + query
        # sublayer 3
        outputs = self.dropout(outputs + self.activation_function(self.feed_forward[i](self.layer_norm(outputs))))
        query = outputs
      output = self.output_linear(query[:, -1, :])
      output = self.layer_norm(output, False)
      dec_output = t.softmax(output, -1)
      return dec_output
    else:
      raise ValueError


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, hidden_dim, num_heads):
    super().__init__()
    assert hidden_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.linear_qs, self.linear_ks, self.linear_vs = self.linear_init()
    self.linear_out = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.activate_function = t.tanh

  def linear_init(self):
    linear_qs = nn.ModuleList([])
    linear_ks = nn.ModuleList([])
    linear_vs = nn.ModuleList([])
    assert self.hidden_dim % self.num_heads == 0
    for i in range(self.num_heads):
      linear_qs.append(nn.Linear(self.hidden_dim, self.hidden_dim // self.num_heads))
      linear_ks.append(nn.Linear(self.hidden_dim, self.hidden_dim // self.num_heads))
      linear_vs.append(nn.Linear(self.hidden_dim, self.hidden_dim // self.num_heads))
    return linear_qs, linear_ks, linear_vs

  def forward(self, query, key, value, original_key=None, sequence_padding=False):
    batch_size = key.size(0)
    heads = [None] * self.num_heads
    for i in range(self.num_heads):
      q_i = self.linear_qs[i](query)  # B * l_q * d_m / num_heads
      k_i = self.linear_ks[i](key)  # B * l_k * d_ / num_heads
      v_i = self.linear_vs[i](value)
      heads[i] = self.scaled_dot_production(q_i, k_i, v_i, original_key, sequence_padding)
    attention = t.cat(heads, -1)
    return self.linear_out(attention)

    # Q = self.activate_function(self.linear_q(query))
    # K = self.activate_function(self.linear_k(key))
    # V = self.activate_function(self.linear_v(value))
    #
    # split_size = self.hidden_dim // self.num_heads
    # Q = t.cat(t.split(Q, split_size, dim=-1), dim=0)  # B*n, L, d_k/n
    # K = t.cat(t.split(K, split_size, dim=-1), dim=0)
    # V = t.cat(t.split(V, split_size, dim=-1), dim=0)
    #
    # attention = self.scaled_dot_production(Q, K, V, original_key, sequence_padding)
    # attention = t.cat(t.split(attention, batch_size, 0), -1)
    # return self.linear_out(attention)

  def scaled_dot_production(self, Q, K, V, ori_key, sequence_padding):
    embed_dim = t.Tensor([self.embed_dim]).to(Q.device)
    attn_weight = t.matmul(Q, K.permute(0, 2, 1)) / t.sqrt(embed_dim)
    # if scale attention weight, it could be easier to train model,
    # but also be inditinguishable after going through softmax function
    # attn_weight = t.matmul(Q, K.permute(0, 2, 1))  # B*n * L_q * L_k
    masks = t.zeros_like(attn_weight)
    if ori_key is not None:
      masks = self.get_padding_masks(Q, ori_key)  # B*n * L_q * L_k
    if sequence_padding:
      masks += self.get_sequence_masks(attn_weight)
    attn_weight = t.softmax(attn_weight + masks, -1)
    return t.matmul(attn_weight, V)

  def get_padding_masks(self, query, key):
    '''
    :param Query: # B*n, L_q, d_k/n
    :param Key: # B*n, L_k, d_m, original key embedding, to get real key sequence length
    :return:
    '''
    neg_large_num = -2 ** 32 + 1
    l_q = query.size(1)
    masks = t.sign(t.sum(t.abs(key), -1)).unsqueeze(1)  # B * 1 * L_k
    masks = masks.repeat(1, l_q, 1)  # B*n * L_q * L_k
    return t.where(masks == 0,
                   t.Tensor([neg_large_num]).to(masks.device),
                   t.Tensor([0]).to(masks.device))

  def get_sequence_masks(self, attn_weight):
    '''
    :param attn_weight: (B*n, L_q, L_q)
    :return:
    '''
    neg_large_num = -2 ** 32 + 1
    masks = t.zeros_like(attn_weight).fill_(neg_large_num)
    return t.tril(masks, diagonal=0)


class LayerNormalization(nn.Module):
  def __init__(self, hidden_dim, is_train=True):
    super(LayerNormalization, self).__init__()
    self.hidden_dim = hidden_dim
    self.alpha = nn.Parameter(t.ones(1, 1))
    self.beta = nn.Parameter(t.zeros(1, 1))
    self.is_train = is_train

  def forward(self, inputs, epsilon=1e-8):
    size = inputs.size()
    # inputs = inputs.view(size[0], -1)
    sigma = t.std(inputs, -1)
    mean = t.mean(inputs, -1)
    output = (inputs - mean.unsqueeze(-1)) / (sigma.unsqueeze(-1) + epsilon)
    if self.is_train:
      output = self.alpha.repeat(size[-2], size[-1]) * output + \
               self.beta.repeat(size[-2], size[-1])
    return output


class PositionalEncoding(nn.Module):
  def __init__(self, embed_dim, rate=0.1):
    super(PositionalEncoding, self).__init__()
    self.model_dim = embed_dim
    self.rate = rate

  def forward(self, inputs):
    batch_size, max_len, _ = inputs.size()
    pe = t.zeros(max_len, self.model_dim).to(inputs.device)
    position = t.arange(1, max_len + 1, dtype=t.float32).unsqueeze(-1).repeat(1, self.model_dim // 2)
    div_term = t.pow(1e5, -t.arange(0, self.model_dim, 2,
                                    dtype=t.float32) / self.model_dim).unsqueeze(0).repeat(max_len, 1)
    pe[:, 0::2] = t.sin(position * div_term)
    pe[:, 1::2] = t.cos(position * div_term)
    pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
    masks = t.sign(t.sum(t.abs(inputs), -1)).unsqueeze(-1).repeat(1, 1, self.model_dim)
    return inputs + self.rate * pe * masks


if __name__ == "__main__":
  transformer = Transformer(64, 1000)
  pass
