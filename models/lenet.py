import torch
import torch.nn as nn


class Transform(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x_input):
        x_node_1_0 = x_input.reshape(-1, *self.out_channels)
        return x_node_1_0


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.node_1_0 = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=2)
        self.node_1_1 = nn.ReLU()
        self.node_2_0 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)
        self.node_2_1 = nn.ReLU()
        self.node_3_0 = Transform([64 * 7 * 7])
        self.node_4_0 = nn.Linear(64 * 7 * 7, 10)
        self.node_4_1 = nn.Softmax(-1)

    def forward(self, x_input):
        x_node_1_0 = self.node_1_0(x_input)
        x_node_1_1 = self.node_1_1(x_node_1_0)
        x_node_2_0 = self.node_2_0(x_node_1_1)
        x_node_2_1 = self.node_2_1(x_node_2_0)
        x_node_3_0 = self.node_3_0(x_node_2_1)
        x_node_4_0 = self.node_4_0(x_node_3_0)
        x_node_4_1 = self.node_4_1(x_node_4_0)
        return x_node_4_1


if __name__ == '__main__':
    model = LeNet()

    data = torch.zeros(42, 1, 28, 28)

    output = model(data)
    assert output.shape == (42, 10)
