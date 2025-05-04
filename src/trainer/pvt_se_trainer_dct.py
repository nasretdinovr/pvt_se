import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from src.common.trainer import BaseTrainer
from src.util.acoustic_utils import mag_phase, get_dct_ideal_ratio_mask, drop_sub_band

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            dist,
            rank,
            config,
            resume: bool,
            model,
            loss_function_mask,
            loss_function_spec,
            loss_function_signal,
            optimizer,
            train_dataloader,
            validation_dataloader
    ):
        super(Trainer, self).__init__(dist,
                                      rank,
                                      config,
                                      resume,
                                      model,
                                      loss_function_mask,
                                      loss_function_spec,
                                      loss_function_signal,
                                      optimizer)

        self.compress_mask = config['loss_function']['mask']['compress']['compress']
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    @staticmethod
    def mask_decompression(pred_RM):
        lim = 9.9
        pred_RM = lim * (pred_RM >= lim) - lim * (pred_RM <= -lim) + pred_RM * (torch.abs(pred_RM) < lim)
        pred_RM = -10 * torch.log((10 - pred_RM) / (10 + pred_RM))
        return pred_RM

    def apply_mask(self, pred_RM, noisy_dct, decompress=True):
        if decompress:
            pred_RM = self.mask_decompression(pred_RM)

        enhanced_dct = pred_RM[..., 0] * noisy_dct[..., 0]
        return enhanced_dct

    def _train_epoch(self, epoch):
        loss_total = 0.0

        i = 0
        desc = f"Training {self.rank}"
        with tqdm(BackgroundGenerator(self.train_dataloader, max_prefetch=20), desc=desc, total=len(self.train_dataloader)) as pgbr:
            for noisy, clean in pgbr:
                self.optimizer.zero_grad()

                noisy = noisy.to(self.rank)
                clean = clean.to(self.rank)

                noisy_dct = self.torch_dct(noisy).unsqueeze(-1)
                clean_dct = self.torch_dct(clean).unsqueeze(-1)
                ground_truth_IRM = get_dct_ideal_ratio_mask(noisy_dct, clean_dct, self.compress_mask)

                with autocast(enabled=self.use_amp):
                    # [B, 1, F, T] => model => [B, 1, F, T]
                    pred_RM = self.model(noisy_dct.permute(0, 3, 1, 2).contiguous())
                    pred_RM = pred_RM.permute(0, 2, 3, 1).contiguous()

                mask_loss = self.loss_function_mask(pred_RM, ground_truth_IRM)

                enhanced_dct = self.apply_mask(pred_RM, noisy_dct, self.compress_mask)
                spec_loss = self.loss_function_spec(enhanced_dct, clean_dct.squeeze(-1))

                enhanced = self.idct(enhanced_dct)
                signal_loss = self.loss_function_signal(enhanced, clean)

                loss = mask_loss+spec_loss+signal_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # if self.rank == 0:
                #     self.writer.add_scalar(f"Loss/loss", loss.item(), i)
                #     self.writer.add_scalar(f"Loss/LR", self.optimizer.param_groups[0]['lr'], i)
                #     i += 1
                #
                # self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*1.01


                loss_total += loss.item()
                pgbr.desc = desc + ' loss = {:7.5f}'.format(loss.item())

            if self.rank == 0:
                self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        speech_loss_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        speech_loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        for i, (noisy, clean, name, speech_type) in tqdm(BackgroundGenerator(enumerate(self.valid_dataloader), 10), desc="Validation"):
            assert len(name) == 1, "The batch size of validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_dct = self.torch_dct(noisy).unsqueeze(-1)
            clean_dct = self.torch_dct(clean).unsqueeze(-1)
            ground_truth_IRM = get_dct_ideal_ratio_mask(noisy_dct, clean_dct, self.compress_mask)  # [B, F, T, 1]

            pred_RM = self.model(noisy_dct.permute(0, 3, 1, 2).contiguous())
            pred_RM = pred_RM.permute(0, 2, 3, 1).contiguous()

            mask_loss = self.loss_function_mask(pred_RM, ground_truth_IRM)

            enhanced_dct = self.apply_mask(pred_RM, noisy_dct, self.compress_mask)
            spec_loss = self.loss_function_spec(enhanced_dct, clean_dct.squeeze(-1))

            enhanced = self.idct(enhanced_dct)
            signal_loss = self.loss_function_signal(enhanced, clean)

            loss = mask_loss+spec_loss+signal_loss

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            """=== === === Visualization === === ==="""
            # Separated Loss
            loss_list[speech_type] += loss
            item_idx_list[speech_type] += 1

            if item_idx_list[speech_type] <= visualization_n_samples:
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)

            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return validation_score_list["No_reverb"]
