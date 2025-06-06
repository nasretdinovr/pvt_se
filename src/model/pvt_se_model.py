import torch
import torch.nn as nn

from src.util.positional_encoding import PositionalEncodingPermute2D


class Model(nn.Module):
    def __init__(self, pe_depth, backbone, head):
        """
        SN-Net model

        Input: [B, 2, F, T]
        Output: [B, 2, F, T]

        Args:

        """
        super().__init__()
        if pe_depth > 0:
            self.pe = PositionalEncodingPermute2D(pe_depth)
        self.backbone = backbone
        self.att_vis = backbone.att_vis
        self.head = head

    def forward(self, x):
            """
            Args:
                x: [B, 2, F, T]

            Returns:
                [B, 2, F, T]
            """
            assert x.dim() == 4
            # # Pad look ahead
            # input = functional.pad(input, [0, self.look_ahead])
            batch_size, n_channels, n_freqs, n_frames = x.size()
            # assert n_channels == 2, f"{self.__class__.__name__} takes complex stft as inputs."
            if hasattr(self, 'pe'):
                x = self.pe(x)
            if self.att_vis:
                x, enc_atts = self.backbone(x)
                x, dec_atts = self.head(x)
                return x, (enc_atts, dec_atts)
            x = self.backbone(x)
            x = self.head(x)
            return x



if __name__ == "__main__":
    import toml
    import sys
    sys.path.append('./../../')
    from src.util.utils import initialize_module, merge_config
    config_name = './../../config/common/pvt_se_train_si_sdr.toml'
    inp = torch.rand((16, 2, 256, 128), device='cuda:0')
    config = toml.load(config_name)

    backbone = initialize_module(config["model"]["backbone"]["path"], args=config["model"]["backbone"]["args"])
    head = initialize_module(config["model"]["head"]["path"], args=config["model"]["head"]["args"])
    model_args = {'backbone': backbone, 'head': head}
    model_args = {**model_args, **config["model"]["args"]}
    model = initialize_module(config["model"]["path"], args=model_args)
    model.to(0)

    o = model(inp)

    print(o)
