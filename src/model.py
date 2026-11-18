import torch
import torch.nn as nn
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BilinearUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(self.up(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bi_upsample: bool = False, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        if bi_upsample:
            self.up_block = BilinearUp(in_channels, out_channels)
        else:
            self.up_block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)


    def forward(self, x):
        return self.up_block(x)


class FiLM(nn.Module):
    def __init__(self, wind_dim: int, channels: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(wind_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channels * 2),
        )

        # Start close to identity modulation
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.channels = channels

    def forward(self, features: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
        params = self.mlp(wind)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        return (1 + gamma) * features + beta


class UNet4(nn.Module):
    def __init__(
        self, 
        in_channels: int = 2, 
        out_channels: int = 1, 
        base_channels: int = 32, 
        depth: int = 4,
        batch_norm=False, 
        dropout_rate: float = 0,
        bi_upsample=False,
        wind_mode: str = "none",
        wind_dim: int = 0,
    ):
        super().__init__()
        self.wind_mode = wind_mode
        if wind_mode.startswith("film"):
            if wind_dim <= 0:
                raise ValueError(f"{wind_mode} requires a positive wind_dim")
            self.film = FiLM(wind_dim, base_channels * 16)
        if wind_mode == "film_grid_cnn":
            self.wind_encoder = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, wind_dim),
                nn.ReLU(inplace=True),
            )

        # 4 encoding levels
        self.enc1 = DoubleConv(in_channels, base_channels, batch_norm=batch_norm)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.enc2 = DoubleConv(base_channels, base_channels * 2, batch_norm=batch_norm)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, batch_norm=batch_norm)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, batch_norm=batch_norm)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout2d(dropout_rate)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, batch_norm=batch_norm)
        self.dropout5 = nn.Dropout2d(dropout_rate)

        # 4 decoding levels
        self.up4 = UpsampleBlock(base_channels * 16, base_channels * 8, bi_upsample, kernel_size=2, stride=2)
        self.dropout6 = nn.Dropout2d(dropout_rate)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, batch_norm=batch_norm)

        self.up3 = UpsampleBlock(base_channels * 8, base_channels * 4, bi_upsample, kernel_size=2, stride=2)
        self.dropout7 = nn.Dropout2d(dropout_rate)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, batch_norm=batch_norm)

        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2, bi_upsample, kernel_size=2, stride=2)
        self.dropout8 = nn.Dropout2d(dropout_rate)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, batch_norm=batch_norm)
        
        self.up1 = UpsampleBlock(base_channels * 2, base_channels, bi_upsample, kernel_size=2, stride=2)
        self.dropout9 = nn.Dropout2d(dropout_rate)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, batch_norm=batch_norm)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _condition(self, wind: torch.Tensor) -> torch.Tensor:
        if self.wind_mode == "film_grid_flat":
            return wind.flatten(1)
        if self.wind_mode == "film_grid_cnn":
            return self.wind_encoder(wind)
        return wind

    # keep this here for now
    def _modulate(self, index: int, features: torch.Tensor, wind: torch.Tensor | None) -> torch.Tensor:
        if wind is None:
            return features
        return self.films[index](features, wind)

    def forward(self, x, wind: torch.Tensor | None = None):
        if self.wind_mode.startswith("film"):
            if wind is None:
                raise ValueError(f"{self.wind_mode} requires a wind tensor")
            wind = self._condition(wind)
        skip1 = self.enc1(x)
        x = self.pool1(skip1)
        x = self.dropout1(x)

        skip2 = self.enc2(x)
        x = self.pool2(skip2)
        x = self.dropout2(x)
        
        skip3 = self.enc3(x)
        x = self.pool3(skip3)
        x = self.dropout3(x)

        skip4 = self.enc4(x)
        x = self.pool4(skip4)
        x = self.dropout4(x)
        
        x = self.bottleneck(x)
        x = self.dropout5(x)
        if wind is not None:
            x = self.film(x, wind)

        x = self.up4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.dropout6(x)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dropout7(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dropout8(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dropout9(x)
        x = self.dec1(x)

        return self.out_conv(x)
