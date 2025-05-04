import math
from functools import lru_cache

import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, cos=True):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        inv_freq = math.pi * torch.pow(2, torch.arange(0, channels).float())
        self.register_buffer('inv_freq', inv_freq)
        self.emb_x = None

    @lru_cache(None)
    def get_pos_emb(self, shape, device):
        batch, x, y, orig_ch = shape
        pos_x = torch.arange(x, device=device).type(self.inv_freq.type()) / x
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = sin_inp_x.cos().unsqueeze(1)
        emb_x = torch.repeat_interleave(emb_x, y, dim=1).unsqueeze(0)
        emb_x = torch.repeat_interleave(emb_x, batch, dim=0)
        return emb_x

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        emb_x = self.get_pos_emb(tensor.shape, tensor.device)
        emb = torch.cat((tensor, emb_x), dim=-1)

        return emb


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)


if __name__ == "__main__":
    x = torch.rand((2, 2, 7, 6))
    pe = PositionalEncodingPermute2D(10)
    out = pe(x)
    print(out)