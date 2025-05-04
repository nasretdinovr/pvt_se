import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from einops import rearrange, repeat
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from src.common.trainer import BaseTrainer
from src.util.acoustic_utils import mag_phase, get_complex_ideal_ratio_mask, drop_sub_band

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

        self.look_ahead = config['args']['look_ahead']
        self.predict_frames = config['args']['predicted_frames']
        self.n_fft = config['acoustic']['n_fft']
        self.hop_length = config['acoustic']['hop_length']
        self.look_ahead_length = self.look_ahead*self.hop_length
        self.clean_length = self.n_fft + (self.predict_frames-1)*self.hop_length
        self.frame_size = config['acoustic']['n_fft']//2 + 1

        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    @staticmethod
    def mask_decompression(pred_cRM):
        lim = 9.9
        pred_cRM = lim * (pred_cRM >= lim) - lim * (pred_cRM <= -lim) + pred_cRM * (torch.abs(pred_cRM) < lim)
        pred_cRM = -10 * torch.log((10 - pred_cRM) / (10 + pred_cRM))
        return pred_cRM

    def apply_mask(self, pred_cRM, noisy_complex, decompress=True):
        if decompress:
            pred_cRM = self.mask_decompression(pred_cRM)
            pred_cRM = pred_cRM.permute(0, 2, 3, 1)

        enhanced_real = pred_cRM[..., 0] * noisy_complex[..., 0] - pred_cRM[..., 1] * noisy_complex[..., 1]
        enhanced_imag = pred_cRM[..., 1] * noisy_complex[..., 0] + pred_cRM[..., 0] * noisy_complex[..., 1]
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
        return enhanced_complex

    def _train_epoch(self, epoch):
        loss_total = 0.0

        i = 0
        desc = f"Training {self.rank}"
        with tqdm(BackgroundGenerator(self.train_dataloader, max_prefetch=40), desc=desc, total=len(self.train_dataloader)) as pgbr:
            for noisy, clean in pgbr:
                self.optimizer.zero_grad()

                start_f, finish_f = self.look_ahead + self.predict_frames, self.look_ahead
                start, finish = self.look_ahead_length + self.clean_length, self.look_ahead_length

                noisy = noisy.to(self.rank)
                clean = clean.to(self.rank)

                noisy_complex = self.torch_stft(noisy)
                noisy_complex_out = noisy_complex[..., -start_f:-finish_f, :]
                clean_complex = self.torch_stft(clean)[..., -start_f:-finish_f, :]
                ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex_out, clean_complex, self.compress_mask)

                # clean_complex = clean_complex.permute(0, 3, 1, 2).contiguous()


                with autocast(enabled=self.use_amp):
                    # [B, F, T, 2] => model => [B, F*T, 2] => [B, F, T, 2]
                    pred_cRM = self.model(noisy_complex)
                    pred_cRM = pred_cRM.reshape(-1, self.frame_size, self.predict_frames, 2).contiguous()
                    b, f, t, c = pred_cRM.shape

                mask_loss = self.loss_function_mask(pred_cRM, ground_truth_cIRM)

                enhanced_complex = self.apply_mask(pred_cRM, noisy_complex_out, self.compress_mask)
                spec_loss = self.loss_function_spec(enhanced_complex, clean_complex.squeeze(-1))

                new_ec = rearrange(enhanced_complex, 'b f t c -> (b t) f () c')
                window = torch.hamming_window(self.n_fft, device=self.rank)

                enhanced = self.istft(new_ec, win_length=None, length=self.n_fft, use_mag_phase=False)

                clean_to_loss = clean[..., -start:-finish].unfold(-1, self.n_fft, self.hop_length).reshape(b*t, -1)
                signal_loss = self.loss_function_signal(enhanced/window, clean_to_loss)

                loss = mask_loss+spec_loss+signal_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                #
                # if self.rank == 0:
                #     self.writer.add_scalar(f"Loss/loss", loss.item(), i)
                #     self.writer.add_scalar(f"Loss/spec_loss", spec_loss.item(), i)
                #     self.writer.add_scalar(f"Loss/LR", self.optimizer.param_groups[0]['lr'], i)
                # i += 1

                # self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*1.01


                loss_total += loss.item()
                pgbr.desc = desc + ' loss = {:5.3f}'.format(loss.item())

            if self.rank == 0:
                self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_spec_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        for i, (noisy, clean, name, speech_type) in tqdm(BackgroundGenerator(enumerate(self.valid_dataloader), 20), desc="Validation"):
            assert len(name) == 1, "The batch size of validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)
            ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex, clean_complex, self.compress_mask)  # [B, F, T, 2]

            i, j = 0, self.predict_frames+self.look_ahead
            start, finish = 0, self.predict_frames
            pred_cRM = torch.zeros_like(noisy_complex, device=self.rank)
            padded_noisy_complex = torch.nn.functional.pad(noisy_complex, [0, 0, 0, self.look_ahead], mode='replicate')
            n_frames = 128
            while j <= padded_noisy_complex.shape[-2]:
                to_model = padded_noisy_complex[..., i:j, :]
                b, f, t, c = to_model.shape
                if t < n_frames:
                    repeat_times = n_frames//t if n_frames%t == 0 else n_frames//t + 1
                    to_model = repeat(to_model, 'b f t c -> b f (t r) c', r=repeat_times)[..., -n_frames:, :]
                    if j + self.predict_frames - i > n_frames:
                        i = j + self.predict_frames - n_frames
                    j = j + self.predict_frames
                else:
                    i, j = i + self.predict_frames, j + self.predict_frames

                pred_cRM_i = self.model(to_model)
                pred_cRM_i = pred_cRM_i.reshape(b, self.frame_size, self.predict_frames, 2).contiguous()
                pred_cRM[..., start:finish, :] = pred_cRM_i
                start, finish = start + self.predict_frames, finish + self.predict_frames

            mask_loss = self.loss_function_mask(pred_cRM, ground_truth_cIRM)

            enhanced_complex = self.apply_mask(pred_cRM, noisy_complex, self.compress_mask)
            spec_loss = self.loss_function_spec(enhanced_complex, clean_complex)

            enhanced = self.istft(enhanced_complex, length=noisy.size(-1), use_mag_phase=False)
            signal_loss = self.loss_function_signal(enhanced, clean)

            loss = mask_loss+spec_loss+signal_loss

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss
            loss_spec_total += spec_loss

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
        self.writer.add_scalar(f"Loss/Validation_spec", loss_spec_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return validation_score_list["No_reverb"]
