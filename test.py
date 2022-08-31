import torch
import torch.nn as nn


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2)

    def forward(self, x):
        return self.conv1(x)


if __name__ == '__main__':
    m = model()
    inp = torch.rand((1, 3, 640, 640))
    oup = m(inp)
    print(oup.shape)
