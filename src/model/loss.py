import functools

import torch
from torch.nn.modules.loss import _Loss

from torchmetrics import SignalNoiseRatio
from torchmetrics.functional.audio.sdr import signal_distortion_ratio
from src.model.stoi_loss import stoi_loss as loss_stoi
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalNoiseRatio
EPS = 1e-8


class NegativeSiSdr(_Loss):
    def __init__(self, size_average, reduce, reduction="mean",
                 zero_mean=True, take_log=True):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        self.sdr_type = 'sisdr'
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_target, target):
        assert target.size() == est_target.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate

        # Step 2. Pair-wise SI-SDR.
        # [batch, 1]
        dot = torch.sum(est_target * target, dim=1, keepdim=True)
        # [batch, 1]
        s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + EPS
        # [batch, time]
        scaled_target = dot * target / s_target_energy

        e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses


class NegativeSNR(_Loss):
    def __init__(self, size_average, reduce, reduction="mean",
                 zero_mean=True, take_log=True):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        self.sdr_type = 'snr'
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_target, target):
        assert target.size() == est_target.size()
        # Step 1. Zero-mean norm

        if self.zero_mean:
            target = target - torch.mean(target, dim=1, keepdim=True)
            est_target = est_target - torch.mean(est_target, dim=1, keepdim=True)

        noise = target - est_target

        snr_value = (torch.sum(target ** 2, dim=1) + EPS) / (torch.sum(noise ** 2, dim=1) + EPS)

        snr_value = 10 * torch.log10(snr_value)
        loss = snr_value.mean()
        return -loss


class NegativeSDR(_Loss):
    def __init__(self, size_average, reduce, reduction="mean",
                 zero_mean=True, take_log=True):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        self.zero_mean = zero_mean
        self.take_log = take_log
        self.use_cg_iter = None
        self.filter_length = 512

    def forward(self, est_target, target):
        assert target.size() == est_target.size()

        sdr_value = signal_distortion_ratio(est_target, target, self.use_cg_iter, self.filter_length, self.zero_mean, EPS)

        loss = sdr_value.mean()
        return -loss

class NegativeSTOI(_Loss):
    def __init__(self, size_average, reduce, reduction="mean",
                 zero_mean=True, take_log=True):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

    def forward(self, est_target, target):
        assert target.size() == est_target.size()

        loss = loss_stoi(est_target, target)

        return loss

class NoLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, est_target, target):
        return 0


def MulLossOutput(loss_function):

    class MulOutput(loss_function):
        def __init__(self, multiplier=1., size_average=None, reduce=None, reduction: str = 'mean', *args, **kwargs):
            super(MulOutput, self).__init__(size_average, reduce, reduction, **kwargs)
            self.multiplier = multiplier

        def forward(self, est_target, target):
            loss = super(MulOutput, self).forward(est_target, target)
            return self.multiplier * loss

    return MulOutput


l1_loss = MulLossOutput(torch.nn.L1Loss)
mse_loss = MulLossOutput(torch.nn.MSELoss)
si_sdr_loss = MulLossOutput(NegativeSiSdr)
no_loss = MulLossOutput(NoLoss)
snr_loss = MulLossOutput(NegativeSNR)
stoi_loss = MulLossOutput(NegativeSTOI)
sdr_loss = MulLossOutput(NegativeSDR)


if __name__ == "__main__":
    loss_snr = snr_loss()
    loss_si_sdr = si_sdr_loss()
    # loss_sdr = sdr_loss()
    # loss_stoi = stoi_loss()

    loss0 = loss_snr(torch.ones((2, 5, 10, 10)), torch.ones((2, 5, 10, 10)) + 6)
    loss3 = loss_si_sdr(torch.ones((2, 5, 10, 10)), torch.ones((2, 5, 10, 10)) + 7)
    # loss1 = loss_sdr(torch.ones((2, 5, 10, 10)), torch.zeros((2, 5, 10, 10)))
    # loss2 = loss_stoi(torch.ones((2, 5, 10, 10)), torch.zeros((2, 5, 10, 10)))
    # print(loss0, loss1, loss2)
    print(loss0, loss3)

    # loss_func = l1_loss(multiplier=0.1)
    # loss_l1 = torch.nn.L1Loss()
    #
    #
    #
    #
    # loss0 = loss_func(torch.ones((2,5,10,10)), torch.zeros((2,5,10,10)))
    # loss1 = loss_l1(torch.ones((2,5,10,10)), torch.zeros((2,5,10,10)))
    #
    # print(loss0, loss1)
    # print(1)

