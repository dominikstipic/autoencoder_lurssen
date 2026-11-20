import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ShipGeometryEncoder(nn.Module):

    def __init__(self, latent_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),       # 256 -> 128
            ConvBlock(32, 64),      # 128 -> 64
            ConvBlock(64, 128),     # 64 -> 32
            ConvBlock(128, 256),    # 32 -> 16
            ConvBlock(256, 512)     # 16 -> 8
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

    def forward(self, x):

        feature_map = self.features(x)

        z = self.fc(
            self.flatten(feature_map)
        )

        return z, feature_map


class ShipGeometryDecoder(nn.Module):

    def __init__(self, latent_dim=512):
        super().__init__()

        self.fc = nn.Linear(
            latent_dim,
            512 * 8 * 8
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512, 256, 4, 2, 1
            ),
            nn.GroupNorm(8, 256),
            nn.GELU(),

            nn.ConvTranspose2d(
                256, 128, 4, 2, 1
            ),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.ConvTranspose2d(
                128, 64, 4, 2, 1
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.ConvTranspose2d(
                64, 32, 4, 2, 1
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.ConvTranspose2d(
                32, 1, 4, 2, 1
            ),

            nn.Sigmoid()
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, 512, 8, 8)

        return self.decoder(x)


class ShipGeometryAutoencoder(nn.Module):

    def __init__(self, latent_dim=512):
        super().__init__()

        self.encoder = ShipGeometryEncoder(latent_dim)
        self.decoder = ShipGeometryDecoder(latent_dim)

    def forward(self, x):

        z, feature_map = self.encoder(x)

        reconstructed = self.decoder(z)

        return reconstructed, z, feature_map