import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, activate_function="PReLU",
                 with_norm=True, final_amplification=1, **kwargs):
        super().__init__()
        self.with_norm = with_norm
        self.activate_function = activate_function
        self.final_amplification = final_amplification

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        )

        if self.with_norm:
            self.norm = nn.LayerNorm(out_channels)

        if activate_function:
            if activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            elif activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif activate_function == "PReLU":
                self.activate_function = nn.ReLU()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:

        """
        x = self.conv(x)
        if self.with_norm:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.activate_function:
            x = self.final_amplification * self.activate_function(x)
        return x
