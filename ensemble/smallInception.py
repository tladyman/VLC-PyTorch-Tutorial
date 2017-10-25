import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallInception(nn.Module):
  def __init__(self):
    super(SmallInception, self).__init__()
    # Channels, hidden units, kernel size
    channels = 3
    self.conv1 = ConvModule(3,96,3,1)

    self.inc1_1 = InceptionModule(96,32,32)
    self.inc1_2 = InceptionModule(64,32,48)
    self.down1 = DownsampleModule(80,80)

    self.inc2_1 = InceptionModule(160,112,48)
    self.inc2_2 = InceptionModule(160,96,64)
    self.inc2_3 = InceptionModule(160,80,80)
    self.inc2_4 = InceptionModule(160,48,96)
    self.down2 = DownsampleModule(144,96)

    self.inc3_1 = InceptionModule(240,176,160)
    self.inc3_2 = InceptionModule(336,176,160)
    # was 7x7, changed to 6 for global pool
    self.pool = nn.AvgPool2d(kernel_size=6, stride=1) 

    self.fc = nn.Linear(336, 10)

  def forward(self, x):
    x = self.conv1(x)

    x = self.inc1_1(x)
    x = self.inc1_2(x)
    x = self.down1(x)

    x = self.inc2_1(x)
    x = self.inc2_2(x)
    x = self.inc2_3(x)
    x = self.inc2_4(x)
    x = self.down2(x)

    x = self.inc3_1(x)
    x = self.inc3_2(x)
    x = self.pool(x)

    x = x.view(x.size(0), 336)
    x = self.fc(x)
    return x


class InceptionModule(nn.Module):
  def __init__(self, I,C1,C3):
    super(InceptionModule, self).__init__()
    # Channels, hidden units, kernel size
    self.conv1 = ConvModule(I,C1,1,1)
    self.conv2 = ConvModule(I,C3,3,1, padding=(1,1))

  def forward(self, x):
    y = self.conv1(x)
    z = self.conv2(x)

    outputs = [y, z]
    return torch.cat(outputs, 1)

class DownsampleModule(nn.Module):
  def __init__(self, I,C3):
    super(DownsampleModule, self).__init__()
    # Channels, hidden units, kernel size
    self.conv1 = ConvModule(I,C3,3,2)
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

  def forward(self, x):
    y = self.conv1(x)
    z = self.pool(x)
    outputs = [y, z]

    return torch.cat(outputs, 1)

class ConvModule(nn.Module):
  def __init__(self, I,C,K,S, padding=(0,0)):
    super(ConvModule, self).__init__()
    # Channels, hidden units, kernel size
    self.conv = nn.Conv2d(I, C, kernel_size=K, stride=S, padding=padding)
    self.batchNorm = nn.BatchNorm2d(C, eps=0.001)

  def forward(self, x):
    x = self.conv(x)
    x = self.batchNorm(x)
    return F.relu(x, inplace=True)
