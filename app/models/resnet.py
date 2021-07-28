import torch.nn as nn


class BaseBlock(nn.Module):
    def __init__(self, c_in, c_out, spatial_downsample):
        super(BaseBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.spatial_downsample = spatial_downsample

        if spatial_downsample:
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.c_in,
                    out_channels=self.c_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=self.c_out),
                nn.ReLU(),
            )

            self.pointwise = nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_out,
                kernel_size=1,
                stride=2,
                padding=0,
            )

        else:
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.c_in,
                    out_channels=self.c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=self.c_out),
                nn.ReLU(),
            )

            self.pointwise = nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_out,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.c_out,
                out_channels=self.c_out,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.c_out),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        identity = self.pointwise(identity)

        x += identity
        x = self.relu(x)

        return x


class BottleNeckBlock(nn.Module):
    def __init__(self):
        super(BottleNeckBlock, self).__init__()
        pass

    def forward(self, x):
        return x


class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes, block_cfg):
        super(ResNet, self).__init__()

        self.block_cfg = block_cfg
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.block_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=3,
                dilation=2,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((56, 56)),
        )

        self.block_1 = self._build_block(
            c_in=64, c_out=64, num_layers=self.block_cfg[0], is_block_1=True
        )
        self.block_2 = self._build_block(
            c_in=64, c_out=128, num_layers=self.block_cfg[1]
        )
        self.block_3 = self._build_block(
            c_in=128, c_out=256, num_layers=self.block_cfg[2]
        )
        self.block_4 = self._build_block(
            c_in=256, c_out=512, num_layers=self.block_cfg[3]
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Conv2d(
            in_channels=512,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = self.avg_pool(x)
        x = self.fc(x)

        return x

    def _build_block(self, c_in, c_out, num_layers, is_block_1=False):
        if is_block_1:
            return nn.Sequential(
                *[
                    BaseBlock(c_in=c_in, c_out=c_out, spatial_downsample=False)
                    if i == 0
                    else BaseBlock(c_in=c_out, c_out=c_out, spatial_downsample=False)
                    for i, _ in enumerate(range(num_layers))
                ]
            )
        else:
            return nn.Sequential(
                *[
                    BaseBlock(c_in=c_in, c_out=c_out, spatial_downsample=True)
                    if i == 0
                    else BaseBlock(c_in=c_out, c_out=c_out, spatial_downsample=False)
                    for i, _ in enumerate(range(num_layers))
                ]
            )


if __name__ == "__main__":
    from torchinfo import summary

    print("Model Summary: \n")
    model = ResNet(input_channels=3, num_classes=200, block_cfg=[3, 4, 6, 3])
    summary(model, input_size=(2, 3, 64, 64))
