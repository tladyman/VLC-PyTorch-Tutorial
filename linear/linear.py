import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        initial = torch.randn(input_channels, output_channels) * 0.01
        self.weight = nn.Parameter(initial)

    def forward(self, x):
        return torch.mm(x, self.weight)


from torch.autograd import Variable

data = Variable(torch.rand(16, 200))  # 16 examples in the minibatch, each with 200 dimensions
module = Linear(200, 100)  # Linear is the module you've defined, taking 200 dimensions to 100 dimensions
output = module(data)  # forward propagate
print(output)  # this should be a tensor with size 16x100
