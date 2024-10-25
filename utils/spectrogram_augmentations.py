import os
import sys
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as T

class SpecAugment():

    def __init__(self, time_mask_param: Optional[int] = 15, freq_mask_param: Optional[int] = 15, W: Optional[int] = 50):
        self.time_mask_param = time_mask_param
        self.W = W
        self.freq_mask_param = freq_mask_param

    def h_poly(self, t):
        tt = t.unsqueeze(-2)**torch.arange(4, device=t.device).view(-1, 1)
        A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
        return A @ tt

    def hspline_interpolate_1D(self, x, y, xs):
        '''
        Input x and y must be of shape (batch, n) or (n)
        '''
        m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
        m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], -1)
        idxs = torch.searchsorted(x[..., 1:], xs)
        dx = (x.take_along_dim(idxs + 1, dim=-1) - x.take_along_dim(idxs, dim=-1))
        hh = self.h_poly((xs - x.take_along_dim(idxs, dim=-1)) / dx)
        return hh[...,0,:] * y.take_along_dim(idxs, dim=-1) \
            + hh[...,1,:] * m.take_along_dim(idxs, dim=-1) * dx \
            + hh[...,2,:] * y.take_along_dim(idxs+1, dim=-1) \
            + hh[...,3,:] * m.take_along_dim(idxs+1, dim=-1) * dx

    def time_warp(self, specs, W=50):
        '''
        Timewarp augmentation

        param:
            specs: spectrogram of size (batch, channel, freq_bin, length)
            W: strength of warp
        '''
        device = specs.device
        batch_size, _, num_rows, spec_len = specs.shape

        mid_y = num_rows // 2
        mid_x = spec_len // 2

        warp_p = torch.randint(W, spec_len - W, (batch_size, ), device=device)

        # Uniform distribution from (0,W) with chance to be up to W negative
        # warp_d = torch.randn(1)*W # Not using this since the paper author make random number with uniform distribution
        warp_d = torch.randint(-W, W, (batch_size, ), device=device)
        x = torch.stack([
            torch.tensor([0], device=device).expand(batch_size), warp_p,
            torch.tensor([spec_len - 1], device=device).expand(batch_size)
        ], 1)
        y = torch.stack([
            torch.tensor([-1.], device=device).expand(batch_size), (warp_p - warp_d) * 2 / (spec_len - 1) - 1,
            torch.tensor([1], device=device).expand(batch_size)
        ], 1)

        # Interpolate from 3 points to spec_len
        xs = torch.linspace(0, spec_len - 1, spec_len, device=device).unsqueeze(0).expand(batch_size, -1)
        ys = self.hspline_interpolate_1D(x, y, xs)

        grid = torch.cat(
            (ys.view(batch_size, 1, -1, 1).expand(-1, num_rows, -1, -1), torch.linspace(
                -1, 1, num_rows, device=device).view(-1, 1, 1).expand(batch_size, -1, spec_len, -1)), -1)

        return torch.nn.functional.grid_sample(specs, grid, align_corners=True)

    def apply(self, spectrogram):
        original_spectrogram = spectrogram.clone()
        time_masking = T.TimeMasking(time_mask_param=self.time_mask_param)
        freq_masking = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)

        augmentation1 = self.time_warp(original_spectrogram.clone(), W=self.W)
        augmentation2 = time_masking(original_spectrogram.clone())
        augmentation3 = freq_masking(original_spectrogram.clone())

        # Collect the original spectrogram and augmentations
        spec_list = [original_spectrogram, augmentation1, augmentation2, augmentation3]

        return spec_list

    def plot_spec(self, spectrogram, prefix: str = "SpecAug", save: Optional[bool] = False):

        spec_list = self.apply(spectrogram)
        titles = [
            "Original", f"Time Warp (W={self.W})", f"Time Mask (M={self.time_mask_param})",
            f"Freq Mask (M={self.freq_mask_param})"
        ]
        spec_list = [spec.squeeze() for spec in spec_list]

        fig, ax = plt.subplots(figsize=(10, 8), nrows=4, sharex=True)
        for i in range(4):
            im = ax[i].imshow(spec_list[i], aspect='auto', origin='lower', cmap='viridis')
            ax[i].set_ylabel('Frequency bins')
            ax[i].set_title(titles[i], fontsize=10)
        plt.xlabel("Time")
        fig.colorbar(im, ax=ax.ravel().tolist(), label='Amplitude')
        if save:
            plt.savefig(prefix)
        plt.show()
