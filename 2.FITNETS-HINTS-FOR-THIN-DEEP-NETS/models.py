import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class FitNet1(nn.Module):
    def __init__(self):
        super(FitNet1, self).__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 16),
            conv_block(16, 16),
            nn.MaxPool2d(2, 2),

            conv_block(16, 32),
            conv_block(32, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2, 2),

            conv_block(32, 48),
            conv_block(48, 48),
            conv_block(48, 64),
            nn.MaxPool2d(8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FitNet2(nn.Module):
    def __init__(self):
        super(FitNet2, self).__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2, 2),

            conv_block(32, 48),
            conv_block(48, 64),
            conv_block(64, 80),
            nn.MaxPool2d(2, 2),

            conv_block(80, 96),
            conv_block(96, 96),
            conv_block(96, 128),
            nn.MaxPool2d(8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6272, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class FitNet3(nn.Module):
    def __init__(self):
        super(FitNet3, self).__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 48),
            conv_block(48, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2, 2),

            conv_block(64, 80),
            conv_block(80, 80),
            conv_block(80, 80),
            conv_block(80, 80),
            nn.MaxPool2d(2, 2),

            conv_block(80, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            nn.MaxPool2d(8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6272, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FitNet4(nn.Module):
    def __init__(self):
        super(FitNet4, self).__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            conv_block(32, 32),
            conv_block(32, 48),
            conv_block(48, 48),
            nn.MaxPool2d(2, 2),

            conv_block(48, 80),
            conv_block(80, 80),
            conv_block(80, 80),
            conv_block(80, 80),
            conv_block(80, 80),
            conv_block(80, 80),
            nn.MaxPool2d(2, 2),

            conv_block(80, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            nn.MaxPool2d(8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6272, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




def conv_block_teacher(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class teacher5(nn.Module):
    def __init__(self):
        super(teacher5, self).__init__()
        self.features = nn.Sequential(
            conv_block_teacher(1, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block_teacher(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block_teacher(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28224, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class teacher5_big(nn.Module):
    def __init__(self):
        super(teacher5_big, self).__init__()
        self.features = nn.Sequential(
            conv_block_teacher(1, 128, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block_teacher(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block_teacher(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56448, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

