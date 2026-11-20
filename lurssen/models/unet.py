import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):

        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(
                half_dim,
                device=t.device
            ) * -emb
        )

        emb = t[:, None] * emb[None, :]

        return torch.cat(
            [emb.sin(), emb.cos()],
            dim=-1
        )

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            )
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            )
        )

        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):

        h = self.conv1(x)

        t = self.time_mlp(t)

        h = h + t[:, :, None, None]

        h = self.conv2(h)

        return h + self.residual(x)

class ConditionalUNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        cond_channels=512,
        base_channels=32,
        time_dim=256
    ):

        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8
        ]

        # Condition projection
        self.condition_projections = nn.ModuleList([
            nn.Conv2d(cond_channels, c, 1)
            for c in channels
        ])

        self.input_conv = nn.Conv2d(
            in_channels,
            channels[0],
            3,
            padding=1
        )

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(channels) - 1):

            self.down_blocks.append(
                ResBlock(
                    channels[i],
                    channels[i],
                    time_dim
                )
            )

            self.downsamples.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    4,
                    stride=2,
                    padding=1
                )
            )

        # Bottleneck
        self.mid1 = ResBlock(
            channels[-1],
            channels[-1],
            time_dim
        )

        self.mid2 = ResBlock(
            channels[-1],
            channels[-1],
            time_dim
        )

        # Decoder
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in reversed(range(len(channels) - 1)):

            self.upsamples.append(
                nn.Conv2d(
                    channels[i + 1],
                    channels[i],
                    3,
                    padding=1
                )
            )

            self.up_blocks.append(
                ResBlock(
                    channels[i] * 2,
                    channels[i],
                    time_dim
                )
            )

        self.output = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(
                channels[0],
                in_channels,
                3,
                padding=1
            )
        )

    def forward(self, x, t, condition):

        t_emb = self.time_embedding(t)

        x = self.input_conv(x)

        skips = []

        # Encoder
        for i in range(len(self.down_blocks)):

            cond = F.interpolate(
                condition,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            x = x + self.condition_projections[i](cond)

            x = self.down_blocks[i](x, t_emb)

            skips.append(x)

            x = self.downsamples[i](x)

        # Bottleneck condition
        cond = F.interpolate(
            condition,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = x + self.condition_projections[-1](cond)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # Decoder
        for i, (up, block) in enumerate(
            zip(self.upsamples, self.up_blocks)
        ):

            skip = skips.pop()

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="nearest"
            )

            x = up(x)

            x = torch.cat(
                [x, skip],
                dim=1
            )

            x = block(x, t_emb)

        return self.output(x)

