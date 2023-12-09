import math

import numpy as np
import torch


class Discriminator(torch.nn.Module):
    # PatchGAN Architecture
    def __init__(self, image_size, depth=5, init_channels=64, label_dim=4):
        super(Discriminator, self).__init__()
        layers = []
        activ = torch.nn.LeakyReLU(0.01)

        # Initial convolution
        initial_conv = torch.nn.Conv2d(
            3, 
            init_channels, 
            kernel_size=4,
            stride=2, 
            padding=1
        )
        layers.append(initial_conv)
        layers.append(activ)

        # Build network
        curr_channels = init_channels
        for i in range(1, depth):
            conv = torch.nn.Conv2d(
                curr_channels, 
                curr_channels*2,
                kernel_size=4, 
                stride=2, 
                padding=1
            )
            curr_channels *= 2
            layers.append(conv)
            layers.append(activ)

        self.network = torch.nn.Sequential(*layers)

        # Final convolutions
        # Image size is halved each convolution
        final_kernel_size = int(image_size / 2**depth)

        # Image classification
        # Probabilistic distribution over image patches
        self.conv_src = torch.nn.Conv2d(
            curr_channels, 
            1,
            kernel_size=3, 
            bias=False
        )

        # Domain classification
        # Probability distributions over each of the labels
        self.conv_cls = torch.nn.Conv2d(
            curr_channels, 
            label_dim,
            kernel_size=final_kernel_size, 
            bias=False
        )

    def forward(self, x):
        out = self.network(x)
        out_src = self.conv_src(out)
        out_cls = self.conv_cls(out)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src, out_cls

    def weight_init(self):
        # He initialisation
        for module in self.modules:
            if isinstance(module, torch.nn.Conv2d):
                kernel_product = np.prod(module.kernel_size)
                n = kernel_product * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2/n))
                if module.bias:
                    module.bias.data.fill_(0)
