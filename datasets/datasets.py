from pathlib import Path
from typing import List, Optional, Union
import sys
import os
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from pandas import DataFrame
from datasets.few_shot_dataset import FewShotDataset
import random
import time
from utils.augmentations import SpecAugment, WaveAugment
import json


class MetaAudioDataset(FewShotDataset):

    def __init__(self, experiment_config,root: Union[Path, str],
                  split: Optional[str] = None,
                  ):
        """
        Build an MetaAudioDataset class to be used for few shot audio classification on meta-audio datasets
        Args:
            root: directory of the main dataset folder - eg /data/BirdClef
            split: a string - can be either "train"/ "test" / "validation" 
        """
        self.experiment_config = experiment_config
        self.root = Path(root)
        self.split = split
        self.multi_segm = experiment_config['multi_segm']
        self.input_type = experiment_config['input_type']
        self.data_df = self.load_specs()
        self.spectograms = self.data_df.filepath.tolist()
        self.class_names = self.data_df.label.unique()
        self.class_to_label = {v: k for k, v in enumerate(self.class_names)}
        self.labels = self.get_labels()
        self.mean, self.std = self.get_normalization_stats()
        self.waveaug_use = self.experiment_config['waveaug_params']['use']
        self.specaug_use = self.experiment_config['specaug_params']['use']

    def __len__(self):
        return len(self.data_df)

    
    def __getitem__(self, item):
        input = np.load(self.data_df['filepath'].iloc[item], allow_pickle=True)
        if self.input_type == "spec":
            if len(input.shape) == 2:
                input = np.expand_dims(input, axis=0)
            input = torch.from_numpy(input)
            normalized_input = self.normalize_spectrogram(input, mean = self.mean, std = self.std)
            normalized_input = normalized_input.unsqueeze(1)
        elif self.input_type == "wav":
            normalized_input = input
        return normalized_input, self.labels[item]
        
    def get_normalization_stats(self):
        norm_stats = np.load(self.root / "norm_stats"/"glob_norm.npy")
        mean = norm_stats[0][0][0]
        std = norm_stats[1][0][0]
        return mean,std
    
    def load_specs(self) -> DataFrame:
        """
        This function will firstly define the desired split based on provided 'split' input. 
        Then it will return a DataFrame with columns filepath,filename,label
        """
        if self.input_type == 'wav' :
            spec_dir = self.root/ "waveforms_npy"
        else:    
            spec_dir = self.root / "features"
        splits_file = np.load(self.root / "splits.npy", allow_pickle=True)

        ## Splits_file is a list of length 3. At splits_file[0] its the training classes, at splits_file[1] the valid_classes etc
        if self.split == 'train':
            labels = splits_file[0]
        elif self.split == 'valid':
            labels = splits_file[1]
        elif self.split == 'test':
            labels = splits_file[2]
        spec_df = pd.DataFrame()
        for label in labels:
            for file in os.listdir(spec_dir / label):
                df_row = {'label': label, 'filename': file, 'filepath': spec_dir / label / file}
                spec_df = pd.concat([spec_df, pd.DataFrame([df_row])])
        spec_df['index_column'] = range(len(spec_df))

        return spec_df

    def get_labels(self) -> List[int]:
        return list(self.data_df.label.map(self.class_to_label))
      
    def normalize_spectrogram(self, spec, mean=None, std=None):
        """
        Normalize a spectrogram or a batch of spectrograms.

        Parameters:
            spec (torch.Tensor): Input spectrogram(s). Shape can be:
                                - [freq_bins, time_bins] for a single segment
                                - [num_of_spec, freq_bins, time_bins] for multiple segments.
            mean (float or torch.Tensor, optional): Precomputed mean value(s). If None, min-max normalization is applied.
            std (float or torch.Tensor, optional): Precomputed std value(s). If None, min-max normalization is applied.

        Returns:
            torch.Tensor: Normalized spectrogram(s) of the same shape as the input.
        """
        # Ensure input is at least 3D for consistent processing
        is_single_segment = spec.dim() == 2  # Check if it's a single spectrogram
        if is_single_segment:
            spec = spec.unsqueeze(0)  # Add batch dimension: [1, freq_bins, time_bins]

        if mean is not None and std is not None:
            # Z-score normalization using provided mean and std
            normalized_spec = (spec - mean) / std
        else:
            # Min-max normalization
            min_val = spec.min()
            max_val = spec.max()

            if max_val == min_val:
                normalized_spec = torch.zeros_like(spec)  # Return a zero tensor if all values are the same
            else:
                normalized_spec = (spec - min_val) / (max_val - min_val)

        # If it was a single segment, remove the batch dimension
        if is_single_segment:
            normalized_spec = normalized_spec.squeeze(0)

        return normalized_spec
    

if __name__ == '__main__':
    with open("experiment_config.json", "r") as f:
        experiment_config = json.load(f)

    data = MetaAudioDataset(experiment_config = experiment_config,root='/data/FSD2018', split='train')
    print((data[0][0].shape))