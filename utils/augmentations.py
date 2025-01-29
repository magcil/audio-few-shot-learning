import os
import sys
import random
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn.functional as FC

from torch_audiomentations import Compose, Gain, PolarityInversion, AddColoredNoise, BandPassFilter, BandStopFilter, HighPassFilter, LowPassFilter, PitchShift, Shift, SpliceOut, TimeInversion, PeakNormalization, AddBackgroundNoise

import numpy as np
import audiomentations as Au
import json

class SpecAugment():

    def __init__(self,
                 experiment_config):
        self.time_mask_param = experiment_config['specaug_params']['mask_param']
        self.W = experiment_config['specaug_params']['W']
        self.freq_mask_param = experiment_config['specaug_params']['mask_param']
        self.freq_num_mask = experiment_config['specaug_params']['num_mask']
        self.time_num_mask = experiment_config['specaug_params']['num_mask']
        self.mask_value = experiment_config['specaug_params']['mask_value']
        self.p = experiment_config['specaug_params']['p']

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

    def apply_augmentations(self, spectrogram):
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


class WaveAugment():
    def __init__(self, experiment_config):
        self.sample_rate = 16000
        self.params = experiment_config['waveaug_params']
        self.dataset = experiment_config['dataset_name']
        self.feature_stats = {
            "FSD2018": {
                "avg_centroid": 1944,
                "avg_bandwidth": 1605,
                "avg_flatness": 0.056
            },
            "nsynth": {
                "avg_centroid": 1294,
                "avg_bandwidth": 961,
                "avg_flatness": 0.224
            },
            "ESC-50-master": {
                "avg_centroid": 1191,
                "avg_bandwidth": 1669,
                "avg_flatness": 0.144
            },
            "BirdClef": {
                "avg_centroid": 3038,
                "avg_bandwidth": 1910,
                "avg_flatness": 0.127
            }
        }
        self.augmentation_composer = self.setup_augmentation_module()

    def setup_augmentation_module(self):
        """
        Prepare the PyTorch audiomentations composer
        """
        def get_adapted_snr(type):
            """
            Adapt min and max SNR db of added noise based on dataset properties

            Args:
                type: min or max
            """
            min_snr = self.params["min_snr_in_db"]
            max_snr = self.params["max_snr_in_db"]
            avg_flatness = self.feature_stats[self.dataset]['avg_flatness']
            adapted_bound = max_snr * (1 - avg_flatness)

            if type == "min":
                adapted_min_snr = random.uniform(min_snr, adapted_bound)
                return adapted_min_snr
            elif type == "max":
                adapted_max_snr = random.uniform(adapted_bound, max_snr)
                return adapted_max_snr

        def get_adapted_lowpass_frequencies(type):
            """
            Adapt lower and upper frequency boundaries of low pass filter based on dataset properties

            Args:
                type: min or max frequency
            """
            avg_centroid = self.feature_stats[self.dataset]['avg_centroid']
            avg_bandwidth = self.feature_stats[self.dataset]['avg_bandwidth']

            if type == "min":
                return avg_centroid
            elif type == "max":
                return avg_centroid + avg_bandwidth / 2

        def get_adapted_highpass_frequencies(type):
            """
            Adapt lower and upper frequency boundaries of high pass filter based on dataset properties

            Args:
                type: min or max frequency
            """
            avg_centroid = self.feature_stats[self.dataset]['avg_centroid']
            avg_bandwidth = self.feature_stats[self.dataset]['avg_bandwidth']

            if type == "min":
                return avg_centroid - avg_bandwidth / 2
            elif type == "max":
                return avg_centroid

        def get_adapted_bandstop_frequencies(type):
            """
            Adapt lower and upper frequency boundaries of band stop filter based on dataset properties

            Args:
                type: min or max frequency
            """
            avg_centroid = self.feature_stats[self.dataset]['avg_centroid']
            avg_bandwidth = self.feature_stats[self.dataset]['avg_bandwidth']

            if type == "min":
                return avg_centroid - avg_bandwidth / 2
            elif type == "max":
                return avg_centroid

        return Compose(
            transforms = [ 
                LowPassFilter(
                    min_cutoff_freq = get_adapted_lowpass_frequencies(type="min"),
                    max_cutoff_freq = get_adapted_lowpass_frequencies(type="min"),
                    mode = "per_example",
                    p = self.params["lowpass_p"],
                    p_mode = None,
                    sample_rate  = self.sample_rate, 
                    target_rate = None,
                    output_type = 'tensor'
                ),
                PitchShift(
                    min_transpose_semitones = self.params["pitchshift_min_transpose_semitones"],
                    max_transpose_semitones =self.params["pitchshift_max_transpose_semitones"],
                    mode = "per_example",
                    p = self.params["pitchshift_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type = 'tensor'
                ),
                Shift(
                    min_shift = self.params["shift_min_shift"],
                    max_shift = self.params["shift_max_shift"],
                    shift_unit = "fraction",
                    rollover = True,
                    mode = "per_example",
                    p = self.params["shift_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type = 'tensor'
                ),
                TimeInversion(
                    mode = "per_example",
                    p = self.params["timeinversion_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type = 'tensor'
                ),
                Gain(
                    min_gain_in_db = self.params["min_gain_in_db"],
                    max_gain_in_db = self.params["max_gain_in_db"],
                    mode = "per_example",
                    p = self.params["gain_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type = 'tensor'
                ),
                AddColoredNoise(
                    min_snr_in_db = get_adapted_snr(type="min"),
                    max_snr_in_db = get_adapted_snr(type="max"),
                    min_f_decay = self.params["noise_min_f_decay"],
                    max_f_decay = self.params["noise_max_f_decay"],
                    mode = "per_example",
                    p = self.params["noise_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type='tensor'
                ),
                HighPassFilter(
                    min_cutoff_freq = get_adapted_highpass_frequencies(type="min"),
                    max_cutoff_freq = get_adapted_highpass_frequencies(type="max"),
                    mode = "per_example",
                    p = self.params["highpass_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type = 'tensor'
                ),
                BandStopFilter(
                    min_center_frequency = get_adapted_bandstop_frequencies(type="min"),
                    max_center_frequency = get_adapted_bandstop_frequencies(type="max"),
                    min_bandwidth_fraction = self.params["bandstop_min_bandwidth_fraction"],
                    max_bandwidth_fraction = self.params["bandstop_max_bandwidth_fraction"],
                    mode = "per_example",
                    p = self.params["bandstop_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type='tensor'
                ),
                SpliceOut(
                    num_time_intervals = self.params["spliceout_num_time_intervals"],
                    max_width = self.params["spliceout_max_width"],
                    mode = "per_example",
                    p = self.params["spliceout_p"],
                    p_mode = None,
                    sample_rate = self.sample_rate,
                    target_rate = None,
                    output_type = 'tensor'
                )
            ]
        )

    def apply_time_masking(self, waveform):
        """
        Apply time masking to the given waveform

        Args:
            waveform: The waveform tensor
        """
        waveform_length = waveform.shape[-1]

        # Calculate the number of samples to mask based on the fraction
        mask_length = int(waveform_length * self.params["timemasking_mask_fraction"])

        # Apply number of masks
        for _ in range(self.params["timemasking_masks"]):
            start_idx = random.randint(0, waveform_length - mask_length)
            waveform[..., start_idx:start_idx + mask_length] = 0

        return waveform

    def apply_time_stretching(self, waveform):
        """
        Apply time stretching to the given waveform

        Args:
            waveform: The waveform tensor
        """
        sox_effects = [
            ["stretch", str(random.uniform(self.params["min_stretch_ratio"], self.params["max_stretch_ratio"]))]
        ]

        if waveform.ndimension() == 3:
            waveform = waveform.squeeze(0)
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 16000, sox_effects)     
        
        # Fix length
        current_length = waveform.shape[-1]  
        if current_length > 80000:
            waveform = waveform[..., :80000]  
        elif current_length < 80000:
            pad_amount = 80000 - current_length
            waveform = FC.pad(waveform, (0, pad_amount))    

        return waveform    

    def apply_augmentations(self, waveform):
        """
        Apply all augmentations to the given waveform

        Args:
            waveform: The waveform tensor
        """
        augmentations = []
        waveform_torch = torch.from_numpy(waveform)
        augmentations.append(waveform_torch)
        waveform_torch = waveform_torch.unsqueeze(0).unsqueeze(0)

        for i in range(self.params['aug_num']):
            # Apply pytorch audiomentations
            augmented_waveform_torch = self.augmentation_composer(waveform_torch, sample_rate=16000)

            # Apply time stretch, if p > 0.5
            timestretch_p = self.params["timestretch_p"]
            if random.uniform(0,1) <= timestretch_p:
                augmented_waveform_torch = self.apply_time_stretching(augmented_waveform_torch)

            # Apply time masking, if p > 0.5
            timemasking_p = self.params["timemasking_p"]
            if random.uniform(0,1) <= timemasking_p:
                augmented_waveform_torch = self.apply_time_masking(augmented_waveform_torch)

            # Resize torch
            if augmented_waveform_torch.ndimension() == 3:
                augmented_waveform_torch = augmented_waveform_torch.squeeze(0)
            
            # Save augmentation
            augmentations.append(augmented_waveform_torch.squeeze(0))
        
        return augmentations


class WaveAugmentOld():
    def __init__(self,experiment_config):
        self.params = experiment_config['waveaug_params']
        self.augmentations = self.define_augmentations()

    def define_augmentations(self):
            augment = Au.Compose(
                [
                    Au.AddGaussianNoise(
                        min_amplitude = self.params['gaussian_min_amplitude'],
                        max_amplitude = self.params['gaussian_max_amplitude'],
                        p = self.params['gaussian_p']
                    ),

                    Au.TimeStretch(
                        min_rate=self.params["timestretch_min_rate"],
                        max_rate=self.params["timestretch_max_rate"],
                        p=self.params["timestretch_p"]
                    ),
        
                    Au.PitchShift(
                        min_semitones=self.params["pitchshift_min_semitones"],
                        max_semitones=self.params["pitchshift_max_semitones"],
                        p=self.params["pitchshift_p"]
                    ),
            
                    Au.Shift(p=self.params["shift_p"]),

                    Au.Aliasing(
                        min_sample_rate=self.params["min_sample_rate"],
                        max_sample_rate=self.params["max_sample_rate"],
                        p=self.params["aliasing_p"]
                    ),

                    Au.LowPassFilter(
                        min_cutoff_freq=self.params["min_cutoff_freq"],
                        max_cutoff_freq=self.params["max_cutoff_freq"],
                        p=self.params["lowpass_p"]
                    ),

                    Au.TimeMask(
                        min_band_part=self.params["min_band_part"],
                        max_band_part=self.params["max_band_part"],
                        p=self.params["timemask_p"]
                    )
                ]
            )
            return augment

    def apply_augmentations(self, waveform):
        augmentations = [torch.from_numpy(waveform)]
        for i in range(self.params['aug_num']):
            augmented_waveform = self.augmentations(samples = waveform, sample_rate = 16000)
            augmentations.append(torch.from_numpy(augmented_waveform))

        return augmentations
    

def preprocessing_and_augmentations(experiment_config, item, test = False):
    read_waveforms = experiment_config['read_waveforms']
    waveaug_use = experiment_config['waveaug_params']['use']
    specaug_use = experiment_config['specaug_params']['use']


    if read_waveforms == False:
        if specaug_use == True:
            augmentation_module = SpecAugment(experiment_config = experiment_config)
            augmented_item_list = augm_module.apply_augmentations(item)
        else:
            augmented_item_list = [item]

    if read_waveforms == True:


        if waveaug_use == True:
            mel_params = {'sr': 16000, 'n_mels':128,'n_fft':1024,'hop_length':512,'power':2.0}
            augmentation_module = WaveAugment(experiment_config = experiment_config)
            augmented_item_list = augmentation_module.apply_augmentations(item)
        else: augmented_item_list = [item]

        ### Now convert to spectrograms
        spec_list = []
        if experiment_config['multi_segm'] == False:
            
            for wav in augmented_item_list:
                spec = mel_spec_function(**mel_params)
                spec_list.append(spec)
        elif experiment_config['multi_segm'] == True:
            for wav in augmented_item_list:
                spec = wav_to_variable_spectrogram(sample = wav, mel_params = mel_params)
                

def mel_spec_function(x, sr, n_mels, n_fft, hop_length, power):
        mel_spec_array = librosa.feature.melspectrogram(y=x,
                                        sr=sr,
                                        n_mels=n_mels,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        power=power)

        log_mel_spec = 20.0 / power * np.log10(mel_spec_array + sys.float_info.epsilon)
        return log_mel_spec



def wav_to_variable_spectrogram(sample,mel_params):
    sample = torch.from_numpy(sample)
    length_s = 5
    expected_size = length_s * mel_params['sr']
    # Collets the raw sample splits
    raw_splits = []

    # If the sample is smaller than expected, we repeat it until we hit the
    #   mark and then trim back if needed
    if sample.shape[0] < expected_size:
        # Calculates the number of repetitions needed of the sample
        multiply_up = int(np.ceil((expected_size) / sample.shape[0]))
        sample = sample.repeat((multiply_up, ))
        # Clips the new sample down as needed
        sample = sample[:expected_size]
        # Store or sample in a list to access later
        raw_splits.append(sample)

    # If the sample is longer than needed, we split it up into its slices
    elif sample.shape[0] >= expected_size:
        starting_index = 0
        while starting_index < sample.shape[0]:
            to_end = sample.shape[0] - starting_index
            # If there more than a full snippet sample still available
            if to_end >= expected_size:
                split = sample[starting_index:(starting_index + expected_size)]
                starting_index += expected_size
                raw_splits.append(split)
            # If we are at the end of our sample
            elif to_end < expected_size:
                # Calculates the number of repetitions needed of the sample
                multiply_up = int(np.ceil((expected_size) / to_end))
                split = sample[starting_index:]
                # Repeats and clips the end sample as needed
                split = sample.repeat((multiply_up, ))[:expected_size]
                starting_index = sample.shape[0]
                raw_splits.append(split)

    mel_splits = []
    # Now need to convert the raw sample into melspectrogram
    # Need to convert to numpy and back again
    for raw in raw_splits:
        mel_spec = mel_spec_function(raw.numpy(), **mel_params)
        mel_spec = torch.from_numpy(mel_spec)
        mel_splits.append(mel_spec)

    x = torch.stack(mel_splits)
    return x
    

        
if __name__ == '__main__':
    with open("../config/experiment_config.json", "r") as f:
        experiment_config = json.load(f)

    augm_module = WaveAugment(experiment_config = experiment_config)
    wav = np.load("/data/ESC-50-master/waveforms_npy/engine/2-106014-A-44.npy", allow_pickle = True)
    aug_list = augm_module.apply_augmentations(wav)
    print(len(aug_list))
    print(aug_list[0].shape)
    print(aug_list[1].shape)
    print(aug_list[2].shape)
    print(aug_list[3].shape)

    import matplotlib.pyplot as plt

    # Assuming `aug_list` contains 4 arrays to plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots

    # Plot each array in a separate subplot
    axes[0, 0].plot(aug_list[0])
    axes[0, 0].set_title('Plot 1')

    axes[0, 1].plot(aug_list[1])
    axes[0, 1].set_title('Plot 2')

    axes[1, 0].plot(aug_list[2])
    axes[1, 0].set_title('Plot 3')

    axes[1, 1].plot(aug_list[3])
    axes[1, 1].set_title('Plot 4')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure to an image file
    plt.savefig('aug_list_plots.png', dpi=300)  # Save with high resolution

    # Optionally show the figure
    plt.show()

    import librosa
    import soundfile as sf
    import numpy as np

    # Assuming aug_list contains your numpy arrays of audio signals
    # Define sample rate (default in librosa is 22050 Hz, but you can customize it)
    sample_rate = 16000

    # Save each array in aug_list as a WAV file
    for i, audio_data in enumerate(aug_list):
        torchaudio.save(f'augmentation_{i}.wav', aug_list[i].unsqueeze(0), sample_rate)
        # # Ensure audio data is a 1D numpy array
        # if len(audio_data.shape) > 1:
        #     audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono

        # # Normalize the audio to avoid clipping
        # max_val = np.max(np.abs(audio_data))
        # if max_val > 1:
        #     audio_data = audio_data / max_val

        # # Save as WAV
        # wav_filename = f"audio_{i + 1}.wav"
        # sf.write(wav_filename, audio_data, samplerate=sample_rate)
        # print(f"Saved: {wav_filename}")


