import torch.nn as nn


class BaseBlock(nn.Module):
    def __init__(self, c_in, c_out, spatial_downsample):
        super(BaseBlock, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.spatial_downsample = spatial_downsample
        self.relu = nn.ReLU()

        if self.spatial_downsample:
            self.base_block = nn.Sequential(
                self.conv_block(
                    c_in=self.c_in, c_out=self.c_out, kernel_size=3, stride=2, padding=1
                ),
                nn.ReLU(),
                self.conv_block(
                    c_in=self.c_out,
                    c_out=self.c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        else:
            self.base_block = nn.Sequential(
                self.conv_block(
                    c_in=self.c_in, c_out=self.c_out, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                self.conv_block(
                    c_in=self.c_out,
                    c_out=self.c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

        if self.c_in != self.c_out:
            self.pointwise_conv = nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_out,
                kernel_size=1,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        identity_layer = x
        x = self.base_block(x)

        if self.c_in == self.c_out:
            x += identity_layer
        else:
            identity_layer = self.pointwise_conv(identity_layer)
            x += identity_layer

        x = self.relu(x)

        return x

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
        )

        return seq_block


class BaseLayer(nn.Module):
    def __init__(self, c_in, c_out, spatial_downsample):
        super(BaseLayer, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.spatial_downsample = spatial_downsample

        self.base_layer = nn.Sequential(
            BaseBlock(
                c_in=self.c_in,
                c_out=self.c_out,
                spatial_downsample=self.spatial_downsample,
            ),
            BaseBlock(c_in=self.c_out, c_out=self.c_out, spatial_downsample=False),
        )

    def forward(self, x):
        x = self.base_layer(x)

        return x


class ResNet(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(ResNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )
        self.layer_1 = BaseLayer(c_in=32, c_out=64, spatial_downsample=True)
        self.layer_2 = BaseLayer(c_in=64, c_out=64, spatial_downsample=False)
        self.layer_3 = BaseLayer(c_in=64, c_out=64, spatial_downsample=False)

        self.layer_4 = BaseLayer(c_in=64, c_out=128, spatial_downsample=True)
        self.layer_5 = BaseLayer(c_in=128, c_out=128, spatial_downsample=False)
        self.layer_6 = BaseLayer(c_in=128, c_out=128, spatial_downsample=False)

        self.layer_7 = BaseLayer(c_in=128, c_out=256, spatial_downsample=True)
        self.layer_8 = BaseLayer(c_in=256, c_out=256, spatial_downsample=False)
        self.layer_9 = BaseLayer(c_in=256, c_out=256, spatial_downsample=False)

        self.layer_10 = BaseLayer(c_in=256, c_out=512, spatial_downsample=True)
        self.layer_11 = BaseLayer(c_in=512, c_out=512, spatial_downsample=False)

        self.gap = nn.AvgPool2d(kernel_size=4)

        self.final_conv = nn.Conv2d(
            in_channels=512, out_channels=self.num_classes, kernel_size=1, stride=1
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        x = self.layer_10(x)
        x = self.layer_11(x)

        x = self.gap(x)
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    print("Model Summary: \n")
    model = ResNet(num_input_channels=3, num_classes=200)
    summary(model, input_size=(2, 3, 64, 64))
