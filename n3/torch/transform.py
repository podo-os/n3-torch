import torch
import torch.nn as nn


class Transform(nn.Module):

    def __init__(self, *out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x_input):
        return x_input.reshape(-1, *self.out_channels)
