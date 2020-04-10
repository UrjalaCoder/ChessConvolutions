import torch.nn as nn
import torch

class ChessNet(nn.Module):

    def __init__(self, in_channels=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=((2, 2)), stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=((2, 2)), stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=((4, 4)), stride=1)
        self.tanh = nn.Tanh()

        self.lossF = nn.MSELoss()
        self.linear = nn.Linear(576, 1)

    def forward(self, input_tensor):
        c1 = self.conv1(input_tensor)
        c2 = self.conv2(self.tanh(c1))
        c3 = self.conv3(self.tanh(c2))
        r = self.tanh(c3)
        _, size1, size2, size3 = r.size()
        # print(c2.size())
        r = r.view(-1, size1*size2*size3)
        l1 = self.linear(r)
        return l1
    
    def loss(self, prediction, target):
        return self.lossF(prediction, target)
