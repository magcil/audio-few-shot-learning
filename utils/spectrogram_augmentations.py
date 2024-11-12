import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as T
import numpy as np


class SpecAugment():

    def __init__(self,
                 experiment_config):
        self.time_mask_param = experiment_config['mask_param']
        self.W = experiment_config['W']
        self.freq_mask_param = experiment_config['mask_param']
        self.freq_num_mask = experiment_config['num_mask']
        self.time_num_mask = experiment_config['num_mask']
        self.mask_value = experiment_config['mask_value']
        self.p = experiment_config['p']

    def frequency_mask(self, spec):
        """
        Apply frequency masking to a spectrogram.
        
        Parameters:
            spec (torch.Tensor): Input spectrogram of shape [batch_size, 1, 128, time].
            F (int): Maximum frequency mask length.
            num_masks (int): Number of frequency masks to apply.
            mask_value (float): Value to fill the masked area with.

        Returns:
            torch.Tensor: Spectrogram with frequency masking applied.
        """
        batch_size, _, _, time = spec.shape
        masked_spec = spec.clone()  # Clone to preserve original spectrogram

        for _ in range(self.freq_num_mask):
            # Randomly choose a frequency band to mask
            f = np.random.randint(1, self.freq_mask_param + 1)  # Mask length should be at least 1
            f0 = np.random.randint(0, 128 - f)  # Starting frequency index

            # Apply the mask
            masked_spec[:, :, f0:f0 + f, :] = self.mask_value

        return masked_spec

    def time_mask(self, spec):
        """
        Apply time masking to a spectrogram.

        Parameters:
            spec (torch.Tensor): Input spectrogram of shape [batch_size, 1, 128, time].
            T (int): Maximum time mask length.
            num_masks (int): Number of time masks to apply.
            mask_value (float): Value to fill the masked area with.
            p (float): Proportion of time steps for maximum time mask length.

        Returns:
            torch.Tensor: Spectrogram with time masking applied.
        """
        batch_size, _, _, time = spec.shape
        masked_spec = spec.clone()  # Clone to preserve original spectrogram

        # Calculate maximum allowed time mask length
        max_time_mask_length = int(self.p * time)

        for _ in range(self.time_num_mask):
            # Randomly choose a time band to mask
            t = np.random.randint(1,
                                  min(self.time_mask_param, max_time_mask_length) +
                                  1)  # Mask length should be at least 1
            t0 = np.random.randint(0, time - t)  # Starting time index

            # Apply the mask
            masked_spec[:, :, :, t0:t0 + t] = self.mask_value

        return masked_spec

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
        augmentation1 = self.time_warp(original_spectrogram.clone(), W=self.W)
        augmentation2 = self.time_mask(original_spectrogram.clone())
        augmentation3 = self.frequency_mask(original_spectrogram.clone())

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
