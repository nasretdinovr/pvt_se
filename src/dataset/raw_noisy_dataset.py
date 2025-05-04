# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import torch
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


def get_clean_name(noisy_name):
    file_id = noisy_name.split('_')[-1]
    test_path = '/'.join(noisy_name.split('/')[:-2])
    return os.path.join(test_path, 'clean', 'clean_fileid_' + file_id)

# def get_clean_name(noisy_name):
#     file_name = noisy_name.split('/')[-1]
#     test_path = '/'.join(noisy_name.split('/')[:-3])
#     return os.path.join(test_path, 'clean_testset_wav', file_name)

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.filenames = glob(f'{path}/**/*.wav', recursive=True)
        self.resample = torchaudio.transforms.Resample(48000, 16000)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        noisy_filename = self.filenames[idx]
        clean_filename = get_clean_name(noisy_filename)
        clean, sr = torchaudio.load(clean_filename)
        if sr == 48000:
            clean = self.resample(clean)
        elif sr != 16000:
            raise ValueError('clean sample rate in not 16000')
        noisy, sr = torchaudio.load(noisy_filename)
        if sr == 48000:
            noisy = self.resample(noisy)
        elif sr != 16000:
            raise ValueError('noisy sample rate in not 16000')

        return clean, noisy


def from_path(noisy_path):
    dataset = NumpyDataset(noisy_path)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=1,
                                       shuffle=False,
                                       pin_memory=False)
