import torch
import torch.nn as nn
from .utils import Flatten

LeNet5 = nn.Sequential(
    nn.Conv2d(3, 20, (5, 5), padding=(2, 2), stride=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, (5, 5), padding=(2, 2), stride=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    Flatten(),
    nn.Linear(50 * 8 * 8, 500),
    nn.ReLU(),
    nn.Linear(500, 10),
)