import os
import torch
import torchaudio


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, epoch=1, filename="wav.scp"):
        super().__init__()
        self.path = path
        self.filename = filename
        self.clean_noisy_paths = None
        self.set_folder(epoch)

    def set_folder(self, epoch):
        with open(os.path.join(self.path, f"{epoch}ep", self.filename), "r") as file:
            clean_noisy_files = file.readlines()
        clean_noisy_files = [tuple(x.strip().split()) for x in clean_noisy_files]
        self.clean_noisy_paths = clean_noisy_files

    def __len__(self):
        return len(self.clean_noisy_paths)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.clean_noisy_paths[idx]
        clean, _ = torchaudio.load(clean_path)
        noisy, _ = torchaudio.load(noisy_path)
        return clean[0], noisy[0]


if __name__ == '__main__':
    import time
    dataset = Dataset("/home/administrator/Data/DNS-Challenge_142h_5ep/", 1)
    cur_time = time.time()
    for i, x in enumerate(dataset):
        if i > 5000:
            print(time.time() - cur_time)
            break
