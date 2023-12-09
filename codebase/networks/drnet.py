import math

import numpy as np
import torch


def conv1x1(channels, out_channels, resize_factor=1):
    conv = torch.nn.Conv2d(
        channels, 
        out_channels * resize_factor,
        kernel_size=1, 
        bias=False
    )

    norm = torch.nn.InstanceNorm2d(
        out_channels * resize_factor, 
        affine=True,
        track_running_stats=True
    )

    return torch.nn.Sequential(conv, norm)


class StandardBlock(torch.nn.Module):
    def __init__(
            self, 
            channels, 
            out_channels, 
            kernel_size=(3, 3),
            dilation=1, 
            residual=True
        ):
        super(StandardBlock, self).__init__()
        self.residual = residual
        self.resize_factor = 1
        self.relu = torch.nn.ReLU(inplace=True)

        # First layer of block
        self.conv1 = torch.nn.Conv2d(
            channels, 
            out_channels,
            kernel_size=kernel_size[0],
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.norm1 = torch.nn.InstanceNorm2d(
            out_channels, 
            affine=True,
            track_running_stats=True
        )

        # Second layer of block
        self.conv2 = torch.nn.Conv2d(
            channels, 
            out_channels,
            kernel_size=kernel_size[1],
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.norm2 = torch.nn.InstanceNorm2d(
            out_channels, 
            affine=True,
            track_running_stats=True
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.residual:
            out += x
        out = self.relu(out)
        return out


class DeepBlock(torch.nn.Module):
    def __init__(
            self, 
            channels, 
            out_channels, 
            kernel_size=(1, 3, 1),
            dilation=1, 
            residual=True
        ):
        super(DeepBlock, self).__init__()
        self.residual = residual
        self.resize_factor = 4
        self.relu = torch.nn.ReLU(inplace=True)

        # First layer of block
        self.conv1 = torch.nn.Conv2d(
            channels, 
            out_channels,
            kernel_size=kernel_size[0],
            bias=False
        )
        self.norm1 = torch.nn.InstanceNorm2d(
            out_channels, 
            affine=True,
            track_running_stats=True
        )

        # Second layer of block
        self.conv2 = torch.nn.Conv2d(
            out_channels, 
            out_channels,
            kernel_size=kernel_size[1],
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.norm2 = torch.nn.InstanceNorm2d(
            out_channels,
            affine=True,
            track_running_stats=True
        )

        # Third layer of block
        self.conv3 = torch.nn.Conv2d(
            out_channels, 
            out_channels * self.resize_factor,
            kernel_size=kernel_size[2], 
            bias=False
        )
        self.norm3 = torch.nn.InstanceNorm2d(
            out_channels * self.resize_factor,
            affine=True,
            track_running_stats=True
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.residual:
            out += x
        out = self.relu(out)
        return out


class DRNET(torch.nn.Module):
    def __init__(
            self, 
            block, 
            downsampling_factor, 
            res_blocks, 
            init_channels, 
            label_dim
        ):
        super(DRNET, self).__init__()
        self.block = block
        self.layer_channels = init_channels
        self.depth = res_blocks
        self.dilation = 1
        self.layers = []
        relu = torch.nn.ReLU(inplace=True)

        # Initial layer
        # Bigger kernel for performance considerations
        # Number of initial channels = RGB channels + label dimensions
        init_conv = torch.nn.Conv2d(
            3+label_dim, 
            self.layer_channels,
            kernel_size=7, 
            stride=1, 
            padding=3,
            bias=False
        )
        norm = torch.nn.InstanceNorm2d(
            self.layer_channels, 
            affine=True,
            track_running_stats=True
        )
        self.layers.append(init_conv)
        self.layers.append(norm)
        self.layers.append(relu)

        # Preparing input if DeepBlock
        if self.block.resize_factor > 1:
            conv = conv1x1(
                self.layer_channels, 
                self.layer_channels,
                resize_factor=self.block.resize_factor
            )
            self.layers.append(conv)
            self.layers.append(relu)

        # Implement residual blocks with dilation
        assert(res_blocks % 2 != 0)

        # In and out channels
        in_channels = self.layer_channels * self.block.resize_factor
        out_channels = self.layer_channels

        for _ in range(res_blocks//2 + 1):
            new_block = self.block(
                in_channels, 
                out_channels,
                dilation=self.dilation
            )
            self.layers.append(new_block)
            self.dilation *= 2

        # Prepare to reduce dilation
        self.dilation = self.dilation // 4
        for _ in range(res_blocks//2):
            new_block = self.block(
                in_channels, 
                out_channels,
                dilation=self.dilation,
                residual=False
            )
            self.layers.append(new_block)
            self.dilation = self.dilation // 2

        # Undo block expansion
        if self.block.resize_factor > 1:
            conv = conv1x1(in_channels, self.layer_channels)
            self.layers.append(conv)
            self.layers.append(relu)

        # Output image
        final_conv = torch.nn.Conv2d(
            init_channels, 
            3,
            kernel_size=7, 
            stride=1, 
            padding=3,
            bias=False
        )
        self.layers.append(final_conv)

        # End with a tanh activation
        self.layers.append(torch.nn.Tanh())

        # Build network
        self.network = torch.nn.Sequential(*self.layers)

        # Initialise weights and biases
        self.weight_init()

    def weight_init(self):
        for module in self.modules:
            # He initialisation
            if isinstance(module, torch.nn.Conv2d):
                kernel_product = np.prod(module.kernel_size)
                n = kernel_product * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2/n))
                if module.bias:
                    module.bias.data.fill_(0)

    def forward(self, x, labels):
        # Extend label information to each of the image pixels
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))

        # Concatenate input pictures and labels
        x = torch.cat([x, labels], dim=1)

        # Compute output
        return self.network(x)
