# Audio few shot CPL + APL experiments

Repository for the paper titled: PROTOTYPICAL CONTRASTIVE LEARNING FOR IMPROVED FEW SHOT AUDIO CLASSIFICATION


## Table of Contents
- [Overview](#overview)
- [Environment Setup](#1-environment-setup)
- [Datasets](#2-datasets)
- [Experiment Config](#3-experiment_config)

## Overview
In this project we study the effect of the combination of Prototypical loss and Contrastive learning for audio few shot classification task. Specaugment module is utlilized producing different versions of the input spectrogram. A self attention transformer is used to concatenate different views of the same input to on enriched representation. We run our experiments in MetaAudio's datasets : ESC-50, FSDKaggle2018, Birdclef2020, Nsynth and a susbset of VoxCeleb1, with the same splits and preprocessing, and encoder backbones as in the original paper for fair comparison. Lastly, the use of Angular loss is proposed.


## 1. Environment Setup

To reproduce the experiments or to perform attacks using any of the algorithms, first create a conda environment with python 3.9 by typing
```bash
conda create -n fewshot_audio python=3.9
```
Then activate the environment
```bash
conda activate fewshot_audio
```
and install the requirements
```bash
pip install -r requirements.txt
```

## 2. Datasets
We used the MetaAudio Datasets for our experiments. For download, preprocessing, check here: 
http://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Dataset%20Processing

**VoxCeleb1** is not now freely available. For this reason :
We used yt-dlp to download VoxCeleb from YouTube. The URL and target segments of the downloaded YT videos were extracted from voxceleb1.csv, the official metadata file of VoxCeleb v1.

After excluding YouTube videos that are no longer available, the dataset consists of 123,756 audio segments from 1,248 unique speakers. Each speaker is identified by both a name (found in the speaker column) and a unique identifier, with the mapping between names and IDs provided in this file.

However, due to an issue during the download process, a significant number of audio segments were saved as empty WAV files. This likely occurred due to extremely large YouTube videos, reducing the number of usable audio segments to 60.184 and the number of unique speakers to 1.246.

Regarding the splits used for training our prototypical networks, a numpy file is provided in the repository of the Meta Audio benchmark. This file includes the baseline train, validation and test splits:

train set: 86.501 audio segments from 873 unique speakers (5 speakers have less than 20 recordings)
validation set: 12.284 audio segments from 125 unique speakers (all speakers have more than 20 recordings)
test set: 24.971 audio segments from from 248 unique speakers (3speakers have less than 20 recordings)
However, due to the downloading issue mentioned earlier, the downloaded audio segments comprise the following sets:

train set: 41.980 audio segments from 873 unique speakers (218 speakers have less than 20 recordings)
validation set: 6.278 audio segments from 125 unique speakers (29 speakers have less than 20 recordings)
test set: 11.926 audio segments from 248 unique speakers (71 speakers have less than 20 recordings)
Filtering process
As it can be observed, many speakers/classes have less than 20 recordings. In order to meet the requirements of the 5-way 5-shot setup of our experiments, it was decided to filter out speakers with less than 20 recordings. This resulted into:

train set: 39.533 audio segments from 655 unique speakers
validation set: 5.943 audio segments from 96 unique speakers
test set: 11.123 audio segments from 177 unique speakers.

**Structure**
Firstly have all your datasets in the same folder. You can change the directory of this folder in  src/train_test.py, data_root variable.
We name each dataset subfolder as : VoxCeleb, nsynth, ESC-50-master, BirdClef, FSD2018.
At each dataset subfolder, there should be 3 basic components. A folder named features, containing the spectrograms organized in a mini-imagenet like structure where each sub folder is named as the corresponding class. Also a file named splits.npy which defines the splits for each dataset and a folder named norm_stats with a file inside named glob_norm.py in which the global normalization stats are stored.

## 3. Experiment Config

Using this repo you can run many different few shot architectures based on ProtoNets.
You can achieve this by knowing how to use the config file 




