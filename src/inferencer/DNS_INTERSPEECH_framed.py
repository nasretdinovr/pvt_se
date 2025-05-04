import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from src.common.inferencer import BaseInferencer
from src.util.acoustic_utils import mag_phase


def cumulative_norm(input):
    eps = 1e-10
    device = input.device
    data_type = input.dtype
    n_dim = input.ndim

    assert n_dim in (3, 4)

    if n_dim == 3:
        n_channels = 1
        batch_size, n_freqs, n_frames = input.size()
    else:
        batch_size, n_channels, n_freqs, n_frames = input.size()
        input = input.reshape(batch_size * n_channels, n_freqs, n_frames)

    step_sum = torch.sum(input, dim=1)  # [B, T]
    step_pow_sum = torch.sum(torch.square(input), dim=1)

    cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
    cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

    entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
    entry_count = entry_count.reshape(1, n_frames)  # [1, T]
    entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

    cum_mean = cumulative_sum / entry_count  # B, T
    cum_var = (cumulative_pow_sum - 2 * cum_mean * cumulative_sum) / entry_count + cum_mean.pow(2)  # B, T
    cum_std = (cum_var + eps).sqrt()  # B, T

    cum_mean = cum_mean.reshape(batch_size * n_channels, 1, n_frames)
    cum_std = cum_std.reshape(batch_size * n_channels, 1, n_frames)

    x = (input - cum_mean) / cum_std

    if n_dim == 4:
        x = x.reshape(batch_size, n_channels, n_freqs, n_frames)

    return x


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super().__init__(config, checkpoint_path, output_dir)

    @staticmethod
    @torch.no_grad()
    def mask_decompression(pred_cRM):
        lim = 9.9
        pred_cRM = lim * (pred_cRM >= lim) - lim * (pred_cRM <= -lim) + pred_cRM * (torch.abs(pred_cRM) < lim)
        pred_cRM = -10 * torch.log((10 - pred_cRM) / (10 + pred_cRM))
        return pred_cRM

    @torch.no_grad()
    def apply_mask(self, pred_cRM, noisy_complex, decompress=True):
        if decompress:
            pred_cRM = self.mask_decompression(pred_cRM)

        enhanced_real = pred_cRM[..., 0] * noisy_complex[..., 0] - pred_cRM[..., 1] * noisy_complex[..., 1]
        enhanced_imag = pred_cRM[..., 1] * noisy_complex[..., 0] + pred_cRM[..., 0] * noisy_complex[..., 1]
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
        return enhanced_complex

    @torch.no_grad()
    def mag(self, noisy, inference_args):
        noisy_complex = self.stft(noisy)
        noisy_mag, noisy_phase = mag_phase(noisy_complex)  # [B, F, T] => [B, 1, F, T]

        enhanced_mag = self.model(noisy_mag.unsqueeze(1)).squeeze(1)

        enhanced = self.istft((enhanced_mag, noisy_phase), length=noisy.size(-1), use_mag_phase=True)
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @torch.no_grad()
    def scaled_mask(self, noisy, inference_args):
        noisy_complex = self.stft(noisy)
        noisy_mag, noisy_phase = mag_phase(noisy_complex)

        # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
        noisy_mag = noisy_mag.unsqueeze(1)
        scaled_mask = self.model(noisy_mag)
        scaled_mask = scaled_mask.permute(0, 2, 3, 1).contiguous()

        enhanced_complex = noisy_complex * scaled_mask
        enhanced = self.istft(enhanced_complex, length=noisy.size(-1), use_mag_phase=False)
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @staticmethod
    def torch_overlap_add(framed, *, frame_step, frame_length=None):
        """Overlap-add ("deframe") a framed signal.

        Parameters other than `framed` are keyword-only.

        Parameters
        ----------
        framed : Tensor of shape `(..., frame_length, n_frames)`.

        frame_step : Overlap to use when adding frames.

        frame_length : Ignored.  Window length and DCT frame length in samples.
            Can be None (default) or same value as passed to `sdct_torch`.

        Returns
        -------
        deframed : Overlap-add ("deframed") signal.
            Tensor of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
        """
        *rest, frame_length2, n_frames = framed.shape
        assert frame_length in {None, frame_length2}
        return torch.nn.functional.fold(
            framed.reshape(-1, frame_length2, n_frames),
            output_size=(((n_frames - 1) * frame_step + frame_length2), 1),
            kernel_size=(frame_length2, 1),
            stride=(frame_step, 1),
        ).reshape(*rest, -1)

    @torch.no_grad()
    def pvt_se_crm_mask(self, noisy, inference_args):

        frame_length = int(inference_args["sub_sample_length"] * inference_args["sr"])
        frame_length -= (frame_length % 2 != 0)
        frame_step = frame_length // 2
        noisy_padded = torch.nn.functional.pad(noisy, (0, frame_step - (noisy.shape[-1] - frame_length) % frame_step))

        framed = noisy_padded.unfold(-1, frame_length, frame_step)

        batch_size, n_frames, _ = framed.shape
        framed = framed.reshape(-1, frame_length)

        noisy_complex = self.stft(framed)

        framed_enhanced = []
        atts = []
        for i in range(n_frames):
            noisy_complex_frame = noisy_complex[batch_size*i:batch_size*(i+1)]
            if self.config['att_vis']:
                pred_cRM, atts_framed = self.model(noisy_complex_frame.permute(0, 3, 1, 2).contiguous())
                atts.append(atts_framed)
            else:
                pred_cRM = self.model(noisy_complex_frame.permute(0, 3, 1, 2).contiguous())
            pred_cRM = pred_cRM.permute(0, 2, 3, 1).contiguous()
            enhanced_complex = self.apply_mask(pred_cRM, noisy_complex_frame, inference_args['compress_mask'])
            enhanced = self.istft(enhanced_complex, length=framed.size(-1), use_mag_phase=False)
            framed_enhanced.append(enhanced)



        framed_enhanced = torch.stack(framed_enhanced, 2)
        window = torch.hamming_window(frame_length).to(framed)
        enhanced = self.torch_overlap_add(framed_enhanced * window.reshape(-1, 1), frame_step=frame_step)
        window_frames = window[:, None].expand(-1, n_frames)
        window_signal = self.torch_overlap_add(window_frames, frame_step=frame_step)

        enhanced = enhanced / window_signal
        enhanced = enhanced[..., :noisy.shape[-1]]

        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        if self.config['att_vis']:
            return enhanced, noisy_complex.detach().squeeze(0).cpu(), atts
        return enhanced

    @torch.no_grad()
    def __call__(self):
        inference_type = self.inference_config["type"]
        assert inference_type in dir(self), f"Not implemented Inferencer type: {inference_type}"

        inference_args = self.inference_config["args"]

        for noisy, name in tqdm(self.dataloader, desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]
            if self.config['att_vis']:
                enhanced, noisy_stft, atts = getattr(self, inference_type)(noisy.to(self.device), inference_args)
                batch_size = noisy.shape[0]
                for i in range(len(atts)):
                    noisy_stft_frame = noisy_stft[batch_size*i:batch_size*(i+1)]
                    os.makedirs(self.enhanced_dir / "atts" / name / f"frame_{i}", exist_ok=True)
                    torch.save(noisy_stft_frame, self.enhanced_dir / "atts" / name / f"frame_{i}" / "noisy_stft.pt")
                    enc_atts, dec_atts = atts[i]
                    for n_block, enc_att in enumerate(enc_atts):
                        torch.save(enc_att, self.enhanced_dir / "atts" / name / f"frame_{i}" / f"enc_block_{n_block}.pt")
                    for n_block, dec_att in enumerate(dec_atts):
                        torch.save(dec_att, self.enhanced_dir / "atts" / name / f"frame_{i}" / f"dec_block_{n_block}.pt")

            else:
                enhanced = getattr(self, inference_type)(noisy.to(self.device), inference_args)

            if abs(enhanced).any() > 1:
                print(f"Warning: enhanced is not in the range [-1, 1], {name}")

            amp = np.iinfo(np.int16).max
            enhanced = np.int16(0.8 * amp * enhanced / np.max(np.abs(enhanced)))

            # clnsp102_traffic_248091_3_snr0_tl-21_fileid_268 => clean_fileid_0
            # name = "clean_" + "_".join(name.split("_")[-2:])
            sf.write(self.enhanced_dir / f"{name}.wav", enhanced, samplerate=self.acoustic_config["sr"])


if __name__ == '__main__':
    a = torch.rand((10, 2, 12545, 166))

