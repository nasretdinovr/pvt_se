import time
from functools import partial
from pathlib import Path
from collections import OrderedDict

import librosa
#import pydct.scipy
#import pydct.torch
import toml
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
#import pydct.scipy
#import pydct.torch
from src.util.acoustic_utils import stft, istft
from src.util.utils import initialize_module, prepare_device, prepare_empty_dir


class BaseInferencer:
    def __init__(self, config, checkpoint_path, output_dir):
        checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        root_dir = Path(output_dir).expanduser().absolute()
        # self.device = prepare_device(torch.cuda.device_count())
        self.config = config['inferencer']['args']

        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:1")


        print("Loading inference dataset...")
        self.dataloader = self._load_dataloader(config["dataset"])
        print("Loading model...")

        self.model, epoch = self._load_model(config["model"], checkpoint_path, self.device)
        self.inference_config = config["inferencer"]

        # self.enhanced_dir = root_dir / f"enhanced_{str(epoch).zfill(4)}"
        self.enhanced_dir = root_dir / f"enhanced"

        prepare_empty_dir([self.enhanced_dir])

        self.acoustic_config = config["acoustic"]
        n_fft = self.acoustic_config["n_fft"]
        hop_length = self.acoustic_config["hop_length"]
        win_length = self.acoustic_config["win_length"]

        self.stft = partial(stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, device=self.device)
        self.istft = partial(istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, device=self.device)
        self.librosa_stft = partial(librosa.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.librosa_istft = partial(librosa.istft, hop_length=hop_length, win_length=win_length)

        # Supported DCT
#        self.torch_dct = partial(pydct.torch.sdct_torch, frame_length=n_fft, frame_step=hop_length)
#        self.idct = partial(pydct.torch.isdct_torch, frame_step=hop_length)
#        self.np_dct = partial(pydct.scipy.sdct, frame_length=n_fft, frame_step=hop_length)
#        self.np_dct = partial(pydct.scipy.sdct, hop_length=hop_length, frame_length=win_length)

        print("Configurations are as follows: ")
        print(toml.dumps(config))
        with open((root_dir / f"{time.strftime('%Y-%m-%d %H:%M:%S')}.toml").as_posix(), "w") as handle:
            toml.dump(config, handle)

    @staticmethod
    def _load_dataloader(dataset_config):
        dataset = initialize_module(dataset_config["path"], args=dataset_config["args"], initialize=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
        )
        return dataloader

    @staticmethod
    def _unfold(input, pad_mode, n_neighbor):
        """
        沿着频率轴，将语谱图划分为多个 overlap 的子频带
        Args:
            input: [B, C, F, T]

        Returns:
            [B, N, C, F, T], F 为子频带的频率轴大小, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}, which should be 4."
        batch_size, n_channels, n_freqs, n_frames = input.size()
        output = input.reshape(batch_size * n_channels, 1, n_freqs, n_frames)
        sub_band_n_freqs = n_neighbor * 2 + 1

        output = functional.pad(output, [0, 0, n_neighbor, n_neighbor], mode=pad_mode)
        output = functional.unfold(output, (sub_band_n_freqs, n_frames))
        assert output.shape[-1] == n_freqs, f"n_freqs != N (sub_band), {n_freqs} != {output.shape[-1]}"

        # 拆分 unfold 中间的维度
        output = output.reshape(batch_size, n_channels, sub_band_n_freqs, n_frames, n_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # permute 本质上与  reshape 可是不同的 ...，得到的维度相同，但 entity 不同啊
        return output

    @staticmethod
    def _load_model(model_config, checkpoint_path, device):

        backbone = initialize_module(model_config["backbone"]["path"], args=model_config["backbone"]["args"])
        head = initialize_module(model_config["head"]["path"], args=model_config["head"]["args"])

        model_args = {'backbone': backbone,  'head': head}
        model_args = {**model_args, **model_config["args"]}

        model = initialize_module(model_config["path"], args=model_args, initialize=True)
        model_checkpoint = torch.load(checkpoint_path, map_location=device)
        model_static_dict = model_checkpoint["model"]
        epoch = model_checkpoint["epoch"]
        print(f"The model breakpoint in tar format is currently being processed, and its epoch is：{epoch}.")

        new_state_dict = OrderedDict()
        for k, v in model_static_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
            
        # load params
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model, model_checkpoint["epoch"]

    def inference(self):
        raise NotImplementedError
