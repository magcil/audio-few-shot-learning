from pathlib import Path
from typing import Callable, List, Optional, Union
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from pandas import DataFrame
from datasets.few_shot_dataset import FewShotDataset
import random


class MetaAudioDataset(FewShotDataset):

    def __init__(self, root: Union[Path, str], split: Optional[str] = None):
        """
        Build an MetaAudioDataset class to be used for few shot audio classification on meta-audio datasets
        Args:
            root: directory of the main dataset folder - eg /data/BirdClef
            split: a string - can be either "train"/ "test" / "validation" 
        """
        self.root = Path(root)
        self.split = split

        self.data_df = self.load_specs()

        self.spectograms = self.data_df.filepath.tolist()

        self.class_names = self.data_df.label.unique()
        self.class_to_label = {v: k for k, v in enumerate(self.class_names)}
        self.labels = self.get_labels()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        spectrogram = np.load(self.data_df['filepath'].iloc[item], allow_pickle=True)
        if len(spectrogram.shape) == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)
        if spectrogram.shape[0] != 1:
            rand_int = random.randint(0, spectrogram.shape[0] - 1)
            spectrogram = spectrogram[rand_int]
            spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = torch.from_numpy(spectrogram)

        return spectrogram, self.labels[item]

    def load_specs(self) -> DataFrame:
        """
        This function will firstly define the desired split based on provided 'split' input. 
        Then it will return a DataFrame with columns filepath,filename,label
        """
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

        return spec_df

    def get_labels(self) -> List[int]:
        return list(self.data_df.label.map(self.class_to_label))


if __name__ == '__main__':
    data = MetaAudioDataset(root='/data/BirdClef', split='train')
    print(data[0])
