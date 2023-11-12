import torch.nn as nn

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


class View(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)