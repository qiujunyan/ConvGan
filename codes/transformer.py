import math
import copy
import torch as t
from torch import nn


def clones(module, N):
  "产生N个相同的层"
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Transformer(nn.Module):
  def __init__(self, embedding, d_ff=64, num_heads=8,
               num_layers=6, dropout=0.1, pad_id=0):
    super(Transformer, self).__init__()
    self.embedding = embedding
    vocab_size, hidden_dim = embedding.weight.size()
    self.pad_id = pad_id
    self.encoder = Encoder(hidden_dim, d_ff, num_heads, num_layers, dropout)
    self.decoder = Decoder(hidden_dim, vocab_size, d_ff, num_heads, num_layers, dropout)

  def forward(self, src, tgt, src_mask, tgt_mask):
    src_embed = self.get_embedding(src)
    tgt_embed = self.get_embedding(tgt)
    enc_output = self.encoder(src_embed, src_mask)
    dec_output = self.decoder(tgt_embed, enc_output, enc_output, tgt_mask, src_mask)
    return dec_output

  def get_embedding(self, index, scale=True):
    embedding = self.embedding(index)
    embed_dim = self.embedding.weight.size(1)
    if scale:
      embedding = embedding * math.sqrt(embed_dim)
    return embedding


class Encoder(nn.Module):
  def __init__(self, hidden_dim, d_ff, num_heads, num_layers, dropout=0.0):
    super(Encoder, self).__init__()
    self.num_layers = num_layers
    self.pe = PositionalEncoding(hidden_dim)
    self.sublayers = clones(SubLayerConnection(hidden_dim, dropout), 2)
    self.self_attn = clones(MultiHeadAttention(hidden_dim, num_heads), num_layers)
    self.feed_forward = clones(FeedForward(hidden_dim, d_ff), num_layers)

  def forward(self, query, mask=None):
    x = self.pe(query)
    for i in range(self.num_layers):
      x = self.sublayers[0](x, lambda x: self.self_attn[i](x, x, x, mask, mask))
      x = self.sublayers[1](x, self.feed_forward[i])
    return x


class Decoder(nn.Module):
  def __init__(self, hidden_dim, vocab_size, d_ff, num_heads, num_layers, dropout=0.0):
    super(Decoder, self).__init__()
    self.num_layers = num_layers
    self.pe = PositionalEncoding(hidden_dim)
    self.sublayers = clones(SubLayerConnection(hidden_dim, dropout), 3)
    self.self_attn = clones(MultiHeadAttention(hidden_dim, num_heads), num_layers)
    self.src_attn = clones(MultiHeadAttention(hidden_dim, num_heads), num_layers)
    self.feed_forward = clones(FeedForward(hidden_dim, d_ff), num_layers)
    self.output_linear = nn.Linear(hidden_dim, vocab_size)

  def forward(self, query, key, value, q_mask=None, k_mask=None):
    x = self.pe(query)
    for i in range(self.num_layers):
      x = self.sublayers[0](x, lambda x: self.self_attn[i](x, x, x, q_mask, q_mask, seq_mask=True))
      x = self.sublayers[1](x, lambda x: self.src_attn[i](x, key, value, q_mask, k_mask))
      x = self.sublayers[2](x, self.feed_forward[i])
    output = self.output_linear(x)
    return t.softmax(output, -1)


class SubLayerConnection(nn.Module):
  def __init__(self, hidden_dim, dropout):
    super(SubLayerConnection, self).__init__()
    self.layer_norm = LayerNormalization(hidden_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, sublayer):
    return inputs + self.dropout(sublayer(self.layer_norm(inputs)))


class MultiHeadAttention(nn.Module):
  def __init__(self, hidden_dim, num_heads):
    super(MultiHeadAttention, self).__init__()
    assert hidden_dim % num_heads == 0
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.active_function = t.tanh
    self.layer_norm = LayerNormalization(hidden_dim, False)
    self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 3)
    self.linear_out = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, query, key, value, q_mask, k_mask, seq_mask=False):
    batch_size = query.size(0)
    #  B * L * d -> B * n * L * d/n
    query, key, value = \
      [linear(x).view(batch_size, -1, self.num_heads,
                      self.hidden_dim // self.num_heads).transpose(1, 2)
       for linear, x in zip(self.linears, (query, key, value))]
    attention = self.scaled_dot_production(query, key, value, q_mask, k_mask, seq_mask)  # B * n * l_q * d /n
    attention = attention.transpose(1, 2).contiguous(). \
      view(batch_size, -1, self.hidden_dim)
    return self.linear_out(attention)

  def scaled_dot_production(self, Q, K, V, q_mask, k_mask, seq_mask=False):
    '''
    :param q_mask: B * l_q
    :param k_mask: B * l_k
    :param seq_mask: if True, mask future tokens
    :return:
    '''
    attn_weight = t.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.hidden_dim)  # B * n * l_q * l_k

    l_q = Q.size(-2)
    l_k = K.size(-2)
    # B * l_k -> B * n * l_q * l_k
    k_mask = k_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, l_q, 1)
    if seq_mask:
      k_mask |= t.triu(t.ones_like(k_mask, dtype=t.bool), diagonal=1)
    k_mask = t.zeros_like(k_mask, dtype=t.float32).masked_fill_(k_mask, -1e32)
    attn_weight = t.softmax(attn_weight + k_mask, -1)

    q_mask = q_mask.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, l_k)
    attn_weight = attn_weight * (~q_mask).int()
    return t.matmul(attn_weight, V)


# Position-wise Feed-Forward Networks
class FeedForward(nn.Module):
  def __init__(self, model_dim, d_ff, dropout=0.1):
    super(FeedForward, self).__init__()
    self.w_1 = nn.Linear(model_dim, d_ff)
    self.w_2 = nn.Linear(d_ff, model_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.w_2(self.dropout(t.relu(self.w_1(x))))


class LayerNormalization(nn.Module):
  def __init__(self, hidden_dim, is_train=True):
    super(LayerNormalization, self).__init__()
    self.hidden_dim = hidden_dim
    if is_train:
      self.alpha = nn.Parameter(t.ones(hidden_dim))
      self.beta = nn.Parameter(t.zeros(hidden_dim))
    self.is_train = is_train

  def forward(self, inputs, epsilon=1e-8):
    size = inputs.size()
    try:
      sigma = t.std(inputs, -1)
    except:
      a = 1
    mean = t.mean(inputs, -1)
    output = (inputs - mean.unsqueeze(-1)) / (sigma.unsqueeze(-1) + epsilon)
    if self.is_train:
      output = self.alpha * output + self.beta
    return output


class PositionalEncoding(nn.Module):
  def __init__(self, embed_dim, max_len=1000, rate=1, dropout=0.1):
    super(PositionalEncoding, self).__init__()
    self.rate = rate
    self.dropout = nn.Dropout(dropout)
    pe = t.zeros(max_len, embed_dim)
    position = t.arange(0., max_len).unsqueeze(1)
    div_term = t.exp(t.arange(0., embed_dim, 2) *
                     -(math.log(10000.0) / embed_dim))

    pe[:, 0::2] = t.sin(position * div_term)  # 偶数列
    pe[:, 1::2] = t.cos(position * div_term)  # 奇数列
    pe = pe.unsqueeze(0)  # [1, max_len, d_model]
    self.register_buffer('pe', pe)

  def forward(self, x):
    try:
      x = x + self.pe[:, :x.size(1)] * self.rate
    except:
      a = 1
    return self.dropout(x)


if __name__ == "__main__":
  transformer = Transformer(64, 1000)
  pass
