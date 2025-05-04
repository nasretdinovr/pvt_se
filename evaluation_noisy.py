import os
import torch
import pandas as pd
import numpy as np

from glob import glob

from tqdm import tqdm
from src.dataset.raw_noisy_dataset import from_path as dataset_from_path
from src.util.metrics import SI_SDR, STOI, WB_PESQ, NB_PESQ


def evaluate(noisy_data_dir, sample_rate):

    dataset = dataset_from_path(noisy_data_dir)
    all_wb_pesq = np.zeros(len(dataset))
    all_nb_pesq = np.zeros(len(dataset))
    all_stoi = np.zeros(len(dataset))
    all_si_sdr = np.zeros(len(dataset))
    it_count = len(dataset)
    for it, (сlean, noisy) in tqdm(enumerate(dataset), total=it_count):
        сlean, noisy = сlean.squeeze().numpy(), noisy.squeeze().numpy()
        wb_pesq = WB_PESQ(сlean, noisy)
        nb_pesq = NB_PESQ(сlean, noisy)
        stoi = STOI(сlean, noisy)
        si_sdr = SI_SDR(сlean, noisy)

        all_wb_pesq[it] = wb_pesq
        all_nb_pesq[it] = nb_pesq
        all_stoi[it] = stoi
        all_si_sdr[it] = si_sdr

    metrics_df = pd.DataFrame(np.stack((glob(f'{noisy_data_dir}/**/*.wav', recursive=True),
                                        all_wb_pesq,
                                        all_nb_pesq,
                                        all_stoi,
                                        all_si_sdr)).transpose(),
                              columns=['filenames', 'wb_pesq', 'nb_pesq', 'stoi', 'si_sdr'])

    csv_dir = '/'.join(noisy_data_dir.split('/')[:-1])+'/'
    metrics_df.to_csv(csv_dir + 'metrics_noisy.csv', index=False)
    metrics_df[['wb_pesq', 'nb_pesq', 'stoi', 'si_sdr']].astype(float).mean().to_csv(csv_dir + 'mean_metrics_noisy.txt', header=False)


if __name__ == '__main__':
    # args = OmegaConf.load(os.path.join('./configs/evaluation_config.yaml'))
    # noisy_data_dir = '/home/administrator/Data/DNS-Challenge/dataset/synthesised/test/audio/noisy'
    # noisy_data_dir = '/media/administrator/Data/DNS-Challenge/datasets/test_set/synthetic/no_reverb/audio/enhanced_PFPL/enhanced'
    # noisy_data_dir = '/media/administrator/Data/demand/test_pfpl/enhanced'
    noisy_data_dir = "/home/administrator/workspace/rauf/PVT_SE/inference_wav/no_rever_b0/enhanced"
    sample_rate = 16000
    evaluate(noisy_data_dir, sample_rate)
