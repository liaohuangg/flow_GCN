import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ThermalUNetMultiTask(nn.Module):
    """Input: power_grid(1,64,64) + layout_mask(1,64,64) + total_power_scalar

    Output:
      - temp_grid: (B,1,64,64) (normalized to [0,1])
      - avg_temp:  (B,1)      (normalized to [0,1])
    """

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()

        self.inc = DoubleConv(in_channels, base)  # 64x64
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))  # bottleneck 8x8

        self.up1 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base * 4, base * 2)

        self.up3 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base * 2, base)

        self.out_grid = nn.Conv2d(base, 1, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base * 8, base * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base * 2, 1),
        )

    def forward(self, power_grid, layout_mask, total_power):
        # power_grid/layout_mask: (B,1,H,W)
        # total_power: (B,1)
        b, _, h, w = power_grid.shape
        total_chan = total_power.view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([power_grid, layout_mask, total_chan], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        d1 = self.up1(x4)
        d1 = torch.cat([x3, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([x1, d3], dim=1)
        d3 = self.conv_up3(d3)

        temp_grid = self.out_grid(d3)

        pooled = self.avg_pool(x4).flatten(1)
        avg_temp = self.fc(pooled)

        return temp_grid, avg_temp
