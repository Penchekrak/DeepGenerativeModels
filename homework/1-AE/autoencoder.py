import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPE = (64, 64)


class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel - 1) // 2, bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return self.act(self.norm(self.conv(x)))


class DefaultAutoEncoder(nn.Module):
    def __init__(self, latent_dimension=64):
        super().__init__()

        self.latent_dimension = latent_dimension

        self.encoder = nn.Sequential(
            Block(1, 16, 3, stride=2),
            Block(16, 32, 3, stride=2),
            Block(32, 32, 3, stride=2),
            Block(32, 32, 3, stride=2),
            Block(32, latent_dimension, 3, stride=1).conv,
        )

        self.decoder = nn.Sequential(
            Block(latent_dimension, 32, 3, upsample=True),
            Block(32, 32, 3, upsample=True),
            Block(32, 32, 3, upsample=True),
            Block(32, 16, 3, upsample=True),
            Block(16, 1, 3).conv,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

    def get_latent_features(self, x):
        self.eval()
        batch_size = x.shape[0]
        return self.encoder(x).view(batch_size, -1)

    def sample(self, n):
        self.eval()
        samples = torch.randn((n, self.latent_dimension, 4, 4), device=next(self.parameters()).device)
        return self.decoder(samples)


class DenoisingBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False,
                 regular_noise_level=0.05):
        super().__init__()
        self.regular_noise_level = regular_noise_level
        self.upsample = upsample
        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel - 1) // 2, bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x = x + torch.randn_like(x) * self.regular_noise_level
        x = self.dropout(x)
        return self.act(self.norm(self.conv(x)))


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, latent_dimension=64, initial_noise_level=0.1, regular_noise_level=0.05):
        super().__init__()

        self.regular_noise_level = regular_noise_level
        self.initial_noise_level = initial_noise_level
        self.latent_dimension = latent_dimension

        self.encoder = nn.Sequential(
            DenoisingBlock(1, 16, 3, stride=2, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(16, 32, 3, stride=2, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(32, 32, 3, stride=2, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(32, 32, 3, stride=2, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(32, self.latent_dimension, 3, stride=1, regular_noise_level=self.regular_noise_level).conv,
        )

        self.decoder = nn.Sequential(
            DenoisingBlock(self.latent_dimension, 32, 3, upsample=True, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(32, 32, 3, upsample=True, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(32, 32, 3, upsample=True, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(32, 16, 3, upsample=True, regular_noise_level=self.regular_noise_level),
            DenoisingBlock(16, 1, 3, regular_noise_level=self.regular_noise_level).conv,
        )

    def forward(self, x):
        if self.training:
            x = self.prepare_input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

    def get_latent_features(self, x):
        self.eval()
        batch_size = x.shape[0]
        return self.encoder(x).view(batch_size, -1)

    def prepare_input(self, x):
        return x + torch.randn_like(x) * self.initial_noise_level

    def sample(self, n):
        self.eval()
        samples = torch.randn((n, self.latent_dimension, 4, 4), device=next(self.parameters()).device)
        return self.decoder(samples)


class DeconvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.conv = nn.ConvTranspose2d(in_features, out_features, kernel, stride=stride,
                                           bias=bias)
        else:
            self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel - 1) // 2,
                                  bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class AdvancedAutoEncoder(nn.Module):
    def __init__(self, latent_dimension=64):
        super().__init__()

        self.latent_dimension = latent_dimension

        self.encoder = nn.Sequential(
            DeconvBlock(1, 16, 3, stride=2),
            DeconvBlock(16, 32, 3, stride=2),
            DeconvBlock(32, 32, 3, stride=2),
            DeconvBlock(32, 32, 3, stride=2),
            DeconvBlock(32, 32, 3, stride=2),
            DeconvBlock(32, latent_dimension, 3, stride=2).conv,
        )

        self.decoder = nn.Sequential(
            DeconvBlock(latent_dimension, 32, 3, stride=2, upsample=True),
            DeconvBlock(32, 32, 3, stride=2, upsample=True),
            DeconvBlock(32, 32, 3, stride=2, upsample=True),
            DeconvBlock(32, 32, 3, stride=2, upsample=True),
            DeconvBlock(32, 16, 3, stride=2, upsample=True),
            DeconvBlock(16, 1, 2, stride=1, upsample=True).conv,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

    def get_latent_features(self, x):
        self.eval()
        batch_size = x.shape[0]
        return self.encoder(x).view(batch_size, -1)

    def sample(self, n):
        self.eval()
        samples = torch.randn((n, self.latent_dimension, 4, 4), device=next(self.parameters()).device)
        return self.decoder(samples)
