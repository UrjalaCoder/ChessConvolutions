import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def calculate_conv_size(in_dimensions, padding, kernel_size, stride):
    in_width, in_height = in_dimensions
    dilation = 1
    # Calculating the height
    out_height = in_height + 2 * padding[0] - dilation * (kernel_size[0] - 1) - 1
    out_height = out_height / stride[0]
    out_height = math.floor(out_height + 1)
    
    # Calculating the width
    out_width = in_width + 2 * padding[1] - dilation * (kernel_size[1] - 1) - 1
    out_width = out_width / stride[1]
    out_width = math.floor(out_width + 1)

    return [out_height, out_width]


class ChessNet(nn.Module):

    def __init__(self, in_channels=5):
        super().__init__()
    
        # 5x8x8 -> 64x4x4
        self.conva1 = nn.Conv2d(in_channels, 16, kernel_size=((3, 3)), stride=1, padding=1)
        self.conva2 = nn.Conv2d(16, 16, kernel_size=((3, 3)), stride=1, padding=1)
        self.conva3 = nn.Conv2d(16, 32, kernel_size=((3, 3)), stride=2, padding=1)
        
        # 32x4x4 -> 64x2x2
        self.convb1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.convb2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.convb3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        
        # 64x2x2 -> 128x1x1
        self.convc1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=1)
        self.convc2 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=1)
        self.convc3 = nn.Conv2d(64, 128, kernel_size=(2, 2), stride=2, padding=0)

        # 128x1x1 -> 128x1x1
        self.convd1 = nn.Conv2d(128, 128, kernel_size=(1, 1))
        self.convd2 = nn.Conv2d(128, 128, kernel_size=(1, 1))
        self.convd3 = nn.Conv2d(128, 128, kernel_size=(1, 1))

        self.tanh = nn.Tanh()

        self.lossF = nn.MSELoss()
        self.linear = nn.Linear(128, 1)

    def forward(self, input_tensor):
        # Result is 32x4x4
        result = F.relu(self.conva1(input_tensor))
        result = F.relu(self.conva2(result))
        result = F.relu(self.conva3(result))
        
        # Result is 64x2x2
        result = F.relu(self.convb1(result))
        result = F.relu(self.convb2(result))
        result = F.relu(self.convb3(result))
        
        # Result is 128x1x1
        result = F.relu(self.convc1(result))
        result = F.relu(self.convc2(result))
        result = F.relu(self.convc3(result))
        
        # Result is 128x1x1
        result = F.relu(self.convd1(result))
        result = F.relu(self.convd2(result))
        result = F.relu(self.convd3(result))

        result = result.view(-1, 128)
        result = self.linear(result)
        return self.tanh(result)
    
    def loss(self, prediction, target):
        return self.lossF(prediction, target)

def main():
    in_width, in_height = 1, 1
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = (0, 0)
    
    r = calculate_conv_size((in_width, in_height), padding, kernel_size, stride)
    print(r)


if __name__ == "__main__":
    main()
