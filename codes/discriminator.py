import torch as t
from torch import nn


class Discriminator(nn.Module):
  def __init__(self, embedding, channels, half_wnd_sizes, pool_ksize,
               seq_len, dropout_prob, device="cuda:0"):
    super(Discriminator, self).__init__()
    self.channels = channels
    self.wnd_sizes = list(map(lambda x: 2 * x + 1, half_wnd_sizes))
    self.embedding = embedding
    self.conv_ksizes = self._init_conv_ksize(embedding.weight.shape[1])
    self.convs = self._init_convs()
    self.out_linear = nn.Linear(seq_len * channels[-1] // 2, 2)
    self.dropout = nn.Dropout(dropout_prob)
    self.activate_function = t.relu
    self.pool = nn.MaxPool2d(pool_ksize)
    self.device = device

    self = self.to(self.device)

  def forward(self, dialogs, neg_ans, ans=None):
    def _get_cross_enctropy_loss(labels, logits):
      acc = t.mean((t.argmax(labels, -1) == t.argmax(logits, -1)).float())
      loss = -t.mean(labels * t.log(1e-10 + logits))
      return loss, acc

    dialogs = self.embedding(dialogs)
    neg_ans = self.embedding(neg_ans)
    if ans is not None: ans = self.embedding(ans)
    inputs, labels = self.data_process(dialogs, neg_ans, ans)
    inputs = inputs.unsqueeze(1)
    bsize = inputs.size(0)
    for layer_convs in self.convs:
      outputs = []
      for conv in layer_convs:
        conv_out = conv(inputs)  # B * C * L * 1
        output_ = self.dropout(self.activate_function(conv_out))
        outputs.append(output_)
      layer_output = (sum(outputs) / len(outputs)).transpose(1, -1)  # B * 1 * L * C
      inputs = self.pool(layer_output)
    output = inputs.view(bsize, -1)
    logits = t.softmax(self.out_linear(output), -1)
    if ans is None:
      return logits[:, 0]
    return _get_cross_enctropy_loss(labels, logits)

  def _init_convs(self):
    convs = nn.ModuleList([])
    for layer, out_channel in enumerate(self.channels):
      conv_ksizes = self.conv_ksizes[layer]
      layer_conv = nn.ModuleList([])
      for i, conv_ksize in enumerate(conv_ksizes):
        row_padding = self.wnd_sizes[i] // 2
        layer_conv.append(nn.Conv2d(1, out_channel, conv_ksize, padding=[row_padding, 0]))
      convs.append(layer_conv)
    return convs

  def _init_conv_ksize(self, model_dim):
    conv_ksizes = []
    channels = [model_dim * 2] + self.channels
    for channel in channels:
      tmp = []
      for wnd_size in self.wnd_sizes:
        tmp.append([wnd_size, channel // 2])
      conv_ksizes.append(tmp)
    return conv_ksizes

  def data_process(self, dialog, neg_ans, ans=None, balance_data=True):
    def _get_onehot(index):
      ret = t.eye(2, device=dialog.device)[index.long()]
      return ret

    if ans is None:
      labels = t.zeros(neg_ans.size(0))
      return t.cat([dialog, neg_ans], 1), _get_onehot(labels)

    data1 = t.cat([dialog, ans], 1)
    try:
      data2 = t.cat([dialog, neg_ans], 1)
    except RuntimeError:
      # this happens when processing wrong answers,
      # one correct answer will be followed by three
      # wrong answers.
      _, repeat_times, seq_len, model_dim = neg_ans.size()
      if balance_data:
        repeat_times = 1
      data2 = t.cat([dialog.repeat(repeat_times, 1, 1),
                     neg_ans.transpose(1, 0)[:repeat_times].
                    contiguous().view(-1, seq_len, model_dim)], 1)
    data = t.cat([data1, data2], 0)
    labels = t.cat([t.ones(data1.size(0)), t.zeros(data2.size(0))], 0)

    # shuffle so that 1s and 0s don't gather to appear
    shuffle_index = t.randperm(data.size(0))
    data = data[shuffle_index, :, :]
    labels = labels[shuffle_index]
    return data, _get_onehot(labels)


if __name__ == "__main__":
  bsize = 10
  seq_len = 32
  model_dim = 64
  inputs = t.rand(bsize, seq_len, model_dim)

  pool_ksize = 2
  channels = [32, 16]
  wnd_sizes = [3, 5, 7, 9]

  discriminator = Discriminator(channels, wnd_sizes)
  discriminator(inputs)
