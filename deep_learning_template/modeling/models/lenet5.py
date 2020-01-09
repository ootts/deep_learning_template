from torch import nn


class LeNet5(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        bsz = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return x
