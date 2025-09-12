import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv (hack to replicate 'SAME' padding from Haiku)
        kernel_size, stride = self.conv.kernel_size[0], self.conv.stride[0]
        assert kernel_size in (1, 3) and stride in (1, 2), f'{kernel_size}, {stride}'
        if kernel_size == 3:
            padding = (1, 1, 1, 1) if stride == 1 else (0, 1, 0, 1)
            x = F.pad(x, padding)
        x = self.conv(x)

        # normalization
        x = self.norm(x)

        # relu
        x = F.relu(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downscale: bool = False):
        super().__init__()

        # downstream layers
        self.layer1 = ResNetLayer(in_channels, out_channels, 3, 1+downscale)
        self.layer2 = ResNetLayer(out_channels, out_channels, 3, 1)
        
        # residual connection
        if downscale:
            self.shortcut = ResNetLayer(in_channels, out_channels, 1, 2)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # downstream layers
        y = self.layer1(x)
        y = self.layer2(y)
        
        # residual connection
        if self.shortcut is not None:
            x = self.shortcut(x)
            
        return F.relu(x + y)

class ResNet(nn.Module):
    def __init__(self, name='resnet20', n_classes=10, n_channels=16):
        super().__init__()
        n_layers = int(name[6:])
        n_blocks_per_stack, r = divmod(n_layers - 2, 6)
        assert r==0, 'num. layers must be 6*n+2 for some integer n'

        self.conv1 = ResNetLayer(3, n_channels)
        
        # three stacks of blocks
        self.stacks = nn.ModuleList()
        output_channels = n_channels
        in_channels = output_channels
        for stack_idx in range(3):
            layers = []
            for block_idx in range(n_blocks_per_stack):
                # Only change channels at first block of stack 2 and 3
                is_transition = stack_idx > 0 and block_idx == 0
                in_channels = output_channels
                if is_transition: output_channels *= 2
                layers.append(ResNetBlock(in_channels, output_channels, is_transition))
            self.stacks.append(nn.Sequential(*layers))

        # fully-connected output layer
        self.fc = nn.Linear(output_channels, n_classes)

        # use mixed precision
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.float()
            else:
                mod.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv1
        x = self.conv1(x)
        
        # three stacks of blocks
        for stack in self.stacks:
            x = stack(x)
        
        # (adaptive) average pooling
        x = torch.mean(x, dim=[2, 3])

        # fully-connected output layer
        x = self.fc(x)
        return x
