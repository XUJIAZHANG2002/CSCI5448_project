# pytorch_resnet50.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------
# Bottleneck Block (ResNet-50)
# -----------------------------------------
class Bottleneck(nn.Module):
    expansion = 4  # channels_out = planes * 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# -----------------------------------------
# ResNet-50
# -----------------------------------------
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 ResNet stages
        self.layer1 = self._make_layer(64,  3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    # create sequential layers
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))

        return nn.Sequential(*layers)

    # forward pass
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # 256
        x = self.layer2(x)  # 512
        x = self.layer3(x)  # 1024
        x = self.layer4(x)  # 2048

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# -----------------------------------------
# X-test
# -----------------------------------------
if __name__ == "__main__":
    model = ResNet50(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("PyTorch ResNet-50 output shape:", out.shape)
