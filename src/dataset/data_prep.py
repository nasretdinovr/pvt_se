import os
import random
import sys
import argparse


import torch
import numpy as np
import toml
from joblib import Parallel, delayed
from scipy import signal
from tqdm import tqdm
from glob import glob

import torchaudio
from torch.utils.data import DataLoader

sys.path.append("/home/administrator/workspace/rauf/PVT_SE")
from src.common.dataset import BaseDataset
from src.util.acoustic_utils import norm_amplitude, tailor_dB_FS, is_clipped, load_wav, subsample


class Dataset(BaseDataset):
    def __init__(self,
                 clean_dataset,
                 clean_dataset_limit,
                 clean_dataset_offset,
                 noise_dataset,
                 noise_dataset_limit,
                 noise_dataset_offset,
                 rir_dataset,
                 rir_dataset_limit,
                 rir_dataset_offset,
                 snr_range,
                 reverb_proportion,
                 silence_length,
                 target_dB_FS,
                 target_dB_FS_floating_value,
                 sub_sample_length,
                 sr,
                 pre_load_clean_dataset,
                 pre_load_noise,
                 pre_load_rir,
                 num_workers,
                 ):
        """
        Dynamic mixing for training

        Args:
            clean_dataset_limit:
            clean_dataset_offset:
            noise_dataset_limit:
            noise_dataset_offset:
            rir_dataset:
            rir_dataset_limit:
            rir_dataset_offset:
            snr_range:
            reverb_proportion:
            clean_dataset: scp file
            noise_dataset: scp file
            sub_sample_length:
            sr:
        """
        super().__init__()
        # acoustic args
        self.sr = sr

        # parallel args
        self.num_workers = num_workers

        clean_dataset_list = glob(f'{clean_dataset}/**/*.wav', recursive=True)
        noise_dataset_list = glob(f'{noise_dataset}/**/*.wav', recursive=True)
        rir_dataset_list = glob(f'{rir_dataset}/**/*.wav', recursive=True)

        clean_dataset_list = self._offset_and_limit(clean_dataset_list, clean_dataset_offset, clean_dataset_limit)
        noise_dataset_list = self._offset_and_limit(noise_dataset_list, noise_dataset_offset, noise_dataset_limit)
        rir_dataset_list = self._offset_and_limit(rir_dataset_list, rir_dataset_offset, rir_dataset_limit)

        if pre_load_clean_dataset:
            clean_dataset_list = self._preload_dataset(clean_dataset_list, remark="Clean dataset")

        if pre_load_noise:
            noise_dataset_list = self._preload_dataset(noise_dataset_list, remark="Noise dataset")

        if pre_load_rir:
            rir_dataset_list = self._preload_dataset(rir_dataset_list, remark="RIR dataset")

        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset_list = noise_dataset_list
        self.rir_dataset_list = rir_dataset_list

        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list

        assert 0 <= reverb_proportion <= 1, "reverberation proportion should be in [0, 1]"
        self.reverb_proportion = reverb_proportion
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length

        self.length = len(self.clean_dataset_list)

    def __len__(self):
        # indexes = list(range(len(self.clean_dataset_list)))
        # random.shuffle(indexes)
        # self.clean_dataset_list = [self.clean_dataset_list[i] for i in indexes]

        return self.length

    def _preload_dataset(self, file_path_list, remark=""):
        waveform_list = Parallel(n_jobs=self.num_workers)(
            delayed(load_wav)(f_path) for f_path in tqdm(file_path_list, desc=remark)
        )
        return list(zip(file_path_list, waveform_list))

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added = load_wav(noise_file, sr=self.sr)
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start:idx_start + target_length]

        return noise_y

    @staticmethod
    def snr_mix(clean_y, noise_y, snr, target_dB_FS, target_dB_FS_floating_value, rir=None, eps=1e-6):
        """
        混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    def __getitem__(self, item):
        clean_file = self.clean_dataset_list[item]
        clean_y = load_wav(clean_file, sr=self.sr)
        clean_y = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr * 4))

        noise_y = self._select_noise_y(target_length=len(clean_y))
        assert len(clean_y) == len(noise_y), f"Inequality: {len(clean_y)} {len(noise_y)}"

        snr = self._random_select_from(self.snr_list)
        use_reverb = bool(np.random.random(1) < self.reverb_proportion)

        noisy_y, clean_y = self.snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            rir=load_wav(self._random_select_from(self.rir_dataset_list), sr=self.sr) if use_reverb else None
        )

        noisy_y = torch.tensor(noisy_y.astype(np.float32))
        clean_y = torch.tensor(clean_y.astype(np.float32))

        return noisy_y, clean_y
        # return clean_y


def get_timesteps_len(speech_timestamps):
    total = 0
    for timesteps in speech_timestamps:
        total += timesteps['end'] - timesteps['start']
    return total


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PVT_SE")
    parser.add_argument("-E", "--epoch", required=True, type=int)
    args = parser.parse_args()

    ep = args.epoch
    batch_size = 16
    total_hours = 142.5
    cfg_path = "/home/administrator/workspace/rauf/PVT_SE/config/common/pvt_se_train_tiny.toml"
    save_path = "/media/administrator/Data/DNS-Challenge_142h_5ep/"
    speech_threshold = 0.4

    os.makedirs(os.path.join(save_path, f"{ep}ep", "noisy"), exist_ok=True)
    os.makedirs(os.path.join(save_path, f"{ep}ep", "clean"), exist_ok=True)
    cfg = toml.load(cfg_path)["train_dataset"]["args"]
    dataset = Dataset(**cfg)

    dataloader = DataLoader(
        dataset=dataset,
        pin_memory=True,
        drop_last=True,
        num_workers=10,
        shuffle=True,
        batch_size=batch_size)

    model, utils = torch.hub.load(github='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=True)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils


    audio_num_samples = int(cfg["sub_sample_length"]*cfg["sr"])

    total_files = int(total_hours * 60 * 60 / cfg["sub_sample_length"])
    file_num = 0
    with open(os.path.join(save_path, f"{ep}ep", "wav.scp"), "w") as file:
        with tqdm(total=total_files) as pgbr:
            for batch in dataloader:
                for noisy, clean in zip(batch[0], batch[1]):
                    num_save_files = clean.shape[0] // audio_num_samples
                    for i in range(0, num_save_files*audio_num_samples, audio_num_samples):
                        speech_timestamps = get_speech_timestamps(clean[i:i + audio_num_samples], model, sampling_rate=cfg["sr"])
                        speech_samples = get_timesteps_len(speech_timestamps)
                        if speech_samples > audio_num_samples * speech_threshold:
                            torchaudio.save(os.path.join(save_path, f"{ep}ep", "noisy", f"noisy_fileid_{file_num}.wav"),
                                            noisy[i:i + audio_num_samples].unsqueeze(0), cfg["sr"])
                            torchaudio.save(os.path.join(save_path, f"{ep}ep", "clean", f"clean_fileid_{file_num}.wav"),
                                            clean[i:i + audio_num_samples].unsqueeze(0), cfg["sr"])
                            file.write(os.path.join(save_path, f"{ep}ep", "noisy", f"noisy_fileid_{file_num}.wav") +
                                       " " + os.path.join(save_path, f"{ep}ep", "clean", f"clean_fileid_{file_num}.wav"))
                            file.write("\n")
                            pgbr.update()
                            file_num += 1
                if file_num >= total_files:
                    break

