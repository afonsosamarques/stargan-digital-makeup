import copy
import math

import numpy as np
import torch


def sampling(
        conv, 
        channels, 
        out_channels, 
        resize_factor, 
        stride=1, 
        dilation=1
    ):
    conv = conv(
        channels, 
        out_channels * resize_factor,
        kernel_size=1, 
        stride=stride, 
        padding=dilation,
        dilation=dilation, 
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
            conv, 
            channels, 
            out_channels, 
            kernel_size=(3, 3),
            dilation=1, 
            stride=1, 
            residual=True, 
            sampling=None
        ):
        super(StandardBlock, self).__init__()
        self.residual = residual
        self.resize_factor = 1
        self.stride = stride
        self.sampling = sampling if sampling is not None else False
        self.relu = torch.nn.ReLU(inplace=True)

        # First layer of block
        self.conv1 = conv(
            channels, 
            out_channels,
            kernel_size=kernel_size[0],
            padding=dilation,
            dilation=dilation,
            stride=stride, 
            bias=False
        )
        self.norm1 = torch.nn.InstanceNorm2d(
            out_channels, 
            affine=True,
            track_running_stats=True
        )

        # Second layer of block
        self.conv2 = conv(
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.residual:
            if self.sampling:
                residual = self.sampling(x)
            else:
                residual = x
            out += residual
        return self.relu(out)


class DeepBlock(torch.nn.Module):
    def __init__(
            self, 
            conv, 
            channels, 
            out_channels, 
            kernel_size=(1, 3, 1),
            dilation=1, 
            stride=1, 
            residual=True, 
            sampling=None
        ):
        super(DeepBlock, self).__init__()
        self.residual = residual
        self.resize_factor = 4
        self.stride = stride
        self.sampling = sampling if sampling is not None else False
        self.relu = torch.nn.ReLU(inplace=True)

        # First layer of block
        self.conv1 = conv(
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
        self.conv2 = conv(
            out_channels, 
            out_channels,
            kernel_size=kernel_size[1],
            padding=dilation,
            dilation=dilation,
            stride=stride,
            bias=False
        )
        self.norm2 = torch.nn.InstanceNorm2d(
            out_channels, 
            affine=True,
            track_running_stats=True
        )

        # Third layer of block
        self.conv3 = conv(
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
            if self.sampling:
                residual = self.sampling(x)
            else:
                residual = x
            out += residual
        return self.relu(out)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Network Towers

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class DownsamplingTower(torch.nn.Module):
    def __init__(
            self, 
            block, 
            channels, 
            dilation, 
            res_blocks, 
            downsampling_blocks
        ):
        super(DownsamplingTower, self).__init__()
        self.block = block
        self.channels = channels
        self.dilation = dilation
        self.downsample_tower = []
        self.out_channels = channels
        self.res_blocks = res_blocks

        # Build downsampling tower blocks
        for _ in range(downsampling_blocks):
            self.out_channels *= 2
            new_block = self.block(
                torch.nn.Conv2d, 
                self.channels, 
                self.out_channels,
                dilation=self.dilation,
                stride=2,
                residual=False
            )
            self.channels = self.out_channels * self.block.resize_factor
            self.downsample_tower.append(new_block)

        # Build non-downsampling tower blocks
        for _ in range(self.res_blocks - downsampling_blocks):
            if self.block.resize_factor > 1:
                downsample_layer = sampling(
                    torch.nn.Conv2d, 
                    self.channels,
                    self.out_channels,
                    self.block.resize_factor,
                    dilation=self.dilation
                )
            else:
                downsample_layer = False

            new_block = self.block(
                torch.nn.Conv2d, 
                self.channels, 
                self.out_channels,
                dilation=self.dilation,
                sampling=downsample_layer
            )
            self.dilation *= 2
            self.channels = self.out_channels * self.block.resize_factor
            self.downsample_tower.append(new_block)

        self.downsample_tower = torch.nn.ModuleList(self.downsample_tower)

    def forward(self, x):
        skip_outputs = []
        out = x
        for module in self.downsample_tower:
            out = module(out)
            # If the module is a block, then save result for skip connection
            if isinstance(module, self.block):
                skip_outputs.append(out)
        return out, skip_outputs


class UpsamplingTower(torch.nn.Module):
    def __init__(
            self, 
            block, 
            channels, 
            dilation, 
            res_blocks, 
            upsampling_blocks
        ):
        super(UpsamplingTower, self).__init__()
        self.block = block
        self.channels = channels
        self.dilation = dilation
        self.upsample_tower = []
        self.out_channels = self.channels // block.resize_factor
        self.res_blocks = res_blocks

        # Build non-upsampling blocks
        for _ in range(self.res_blocks - upsampling_blocks):
            self.dilation = self.dilation // 2

            new_block = self.block(
                torch.nn.Conv2d, 
                self.channels, 
                self.out_channels,
                dilation=self.dilation,
                residual=False
            )
            self.channels = self.out_channels * self.block.resize_factor
            self.upsample_tower.append(new_block)

        assert(self.dilation == 1)

        # Build upsampling blocks
        for _ in range(upsampling_blocks):
            self.out_channels = self.out_channels // 2

            # Twice as many channels due to skip connection
            new_block = self.block(
                torch.nn.ConvTranspose2d,
                self.channels*2,
                self.out_channels,
                dilation=self.dilation,
                stride=2,
                residual=False
            )
            self.channels = self.out_channels * self.block.resize_factor
            self.upsample_tower.append(new_block)

        self.upsample_tower = torch.nn.ModuleList(self.upsample_tower)

    def forward(self, x, skip_outputs):
        blocks_left = self.n_blocks
        out = x
        for module in self.upsample_tower:
            # If the module is a block, then pass on result from skip connection
            if isinstance(module, self.block):
                if module.get_stride() > 1:
                    # Re-size out to intended size
                    if (skip_outputs[blocks_left-1].size(2) - out.size(2)) != 0:
                        dims = skip_outputs[blocks_left-1].size(2)
                        out = torch.nn.functional.interpolate(
                            out, 
                            [dims, dims],
                            mode='bilinear',
                            align_corners=False
                        )
                    out = torch.cat((out, skip_outputs[blocks_left-1]), dim=1)
                blocks_left -= 1
            out = module(out)

        # If it's an odd size, then upsample one last time
        if (out.size(2) % 2 != 0):
            out = torch.nn.functional.interpolate(
                out, 
                [out.size(2)+1, out.size(2)+1],
                mode='bilinear',
                align_corners=False
            )
        
        return out


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Network

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class DRUNET(torch.nn.Module):
    def __init__(
            self, 
            block, 
            res_blocks, 
            downsampling_factor, 
            init_channels, 
            label_dim
        ):
        super(DRUNET, self).__init__()
        self.block = block
        self.channels = init_channels
        self.tower_depth = res_blocks
        self.dilation = 1
        strided_blocks = int(math.log(downsampling_factor)/math.log(2))
        assert(self.tower_depth >= strided_blocks)

        #
        # Initial layer
        self.init_conv = torch.nn.Conv2d(
            3+label_dim, 
            self.channels,
            kernel_size=7, 
            stride=1, 
            padding=3,
            bias=False
        )
        self.norm = torch.nn.InstanceNorm2d(
            self.channels, 
            affine=True,
            track_running_stats=True
        )
        self.relu = torch.nn.ReLU(inplace=True)

        #
        # Build downsampling tower
        self.downsample_tower = DownsamplingTower(
            self.block, 
            self.channels,
            dilation=self.dilation,
            res_blocks=res_blocks,
            downsampling_blocks=strided_blocks
        )
        self.channels *= self.block.resize_factor * (2**strided_blocks)

        #
        # Bridge, linking downsampling and upsampling towers
        self.dilation = 2**(self.tower_depth-strided_blocks)
        self.bridge = self.block(
            torch.nn.Conv2d, 
            self.channels, 
            self.channels // self.block.resize_factor,
            dilation=self.dilation
        )

        #
        # Build upsampling tower
        # Bridge does not change inputs
        self.upsample_tower = UpsamplingTower(
            self.block, 
            self.channels,
            dilation=self.dilation,
            res_blocks=res_blocks,
            upsampling_blocks=strided_blocks
        )
        self.channels = self.channels // (2**strided_blocks)

        # Output image
        self.final_conv = torch.nn.Conv2d(
            init_channels*2, 
            3,
            kernel_size=7, 
            stride=1, 
            padding=3,
            bias=False
        )
        self.tanh = torch.nn.Tanh()

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
        out = torch.cat([x, labels], dim=1)

        # First convolution
        out = self.init_conv(out)
        out = self.norm(out)
        out = self.relu(out)
        skip = copy.deepcopy(out)

        # Downsampling Tower
        out, skip_outputs = self.downsample_tower(out)

        # Bridge connection
        out = self.bridge(out)

        # Upsampling Tower
        out = self.upsample_tower(out, skip_outputs)

        # Output image
        out = torch.cat([out, skip], dim=1)
        out = self.final_conv(out)
        return self.tanh(out)
