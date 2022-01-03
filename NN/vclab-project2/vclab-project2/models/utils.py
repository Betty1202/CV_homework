import torch
import torch.nn as nn

def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)