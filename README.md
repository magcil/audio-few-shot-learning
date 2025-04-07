# Audio few shot CPL + APL experiments

Repository for the paper titled: PROTOTYPICAL CONTRASTIVE LEARNING FOR IMPROVED FEW SHOT AUDIO CLASSIFICATION


## Table of Contents
- [Overview](#overview)
- [Environment Setup](#1-environment-setup)
- [Datasets](#2-datasets)
- [ExperimentConfig](#3-experiment_config)
- [Scenarios](#4-scenarios)
- [ModelConfig](#5-model_config)
- [Hyperparameters](#6-hyperparameters)
- [Run](#7-run)

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

## 3. ExperimentConfig

Using this repo you can run many different few shot architectures based on ProtoNets.
You can achieve this by knowing how to use the config file.

**experiment_config.json:**
```
{
    "encoder_name":"Hybrid",
    "dataset_name":"FSD2018",
    "use_attention":true,
    "use_contrastive":true,
    "input_type":"spec",
    "n_way_train":5,
    "n_way_validation":5,
    "n_way_test":5,
    "n_shot_train":5,
    "n_shot_validation":5,
    "n_shot_test":5,
    "n_query_train":5,
    "n_query_validation":5,
    "n_query_test":5,
    "train_query_augmentations":true,
    "validation_query_augmentations":true,
    "test_query_augmentations":true,
    "lr":0.0007,
    "loss":
    {
        "l_param": 2.022308,
    
        "cpl":
        {
            "use": true,
            "m_param":5,
            "t_param":9.2361
        },
        "angular":
        {
            "use":false,
            "angle": 0,
            "prototypes_as_anchors": true
        }
    },
    
    "num_epochs":200,
    "multi_segm":true,
    "tie_strategy": "",
    "relation_head":false,
    "n_training_tasks":100,
    "n_testing_tasks":2000,
    
    "device": "cuda",
    "gpu_index":0,
    "scheduler_milestones":[20,40,60],
    "scheduler_gamma":0.4482,
    "patience":70,
    "experiment_folder":"FSD_PROTO_PLAIN_CPL",
    "normalize_prototypes": true,
    "project_prototypes":true,
    "specaug_params":
        {
        "use":false,
        "mask_param":16,
        "W":22,
        "num_mask":1,
        "mask_value":0,
        "p":0.282},
    "waveaug_params": {"use":false,
            "aug_num": 3,
            "min_gain_in_db": -6,
            "max_gain_in_db": 6,
            "gain_p": 0.5,
            "min_snr_in_db": 10,
            "max_snr_in_db": 25,
            "noise_min_f_decay": -2,
            "noise_max_f_decay": 2,
            "noise_p": 0.5,
            "bandstop_min_bandwidth_fraction": 0.5,
            "bandstop_max_bandwidth_fraction": 1,
            "bandstop_p": 0.5,
            "highpass_p": 0.3,
            "lowpass_p": 0.5,
            "pitchshift_min_transpose_semitones": -4,
            "pitchshift_max_transpose_semitones": 4,
            "pitchshift_p": 0.5,
            "shift_min_shift": -0.5,
            "shift_max_shift": 0.5,
            "shift_p": 0.5,
            "spliceout_num_time_intervals": 8,
            "spliceout_max_width": 400,
            "spliceout_p": 0.5,
            "timeinversion_p": 0,
            "min_stretch_ratio": 0.9,
            "max_stretch_ratio": 1.1,
            "timestretch_p": 0,
            "timemasking_masks": 5,
            "timemasking_mask_fraction": 0.01,
            "timemasking_p": 0.5
        }
    }
```
Now lets break down every arguement: 

- "encoder_name": The backbone model, can be "Hybrid" for CRNN or "CNN" for a plain 4-64 convnet without the RNN layer
- "dataset_name": Should be the same with each dataset folder name. This defines the dataset that the experiment will run. Can be ESC-50-master, voxceleb, FSD2018, BirdClef, nsynth.
- "use_attention" : Can be true or false. If true a self attention layer will be used, which gets the list of features comming from augmented versions of the same spectrogram, and concatenates them to one vector. If not, the attention is skipped and the augmentations just enrich the few shot batch with more data
- "use_contrastive": if use contrastive is true, then one of the two contrastive losses can be used along with the projection head. If it's false, only prototypical loss is used during training.
- "input_type": can be spec or wav.
- "n_way_train", "n_way_validation", "n_way_test", "n_shot_train", "n_shot_validation", "n_shot_test", "n_query_train", "n_query_validation", "n_query_test": n way - k shot parameters for few shot training, pretty sel explanatory.
- "train_query_augmentations", "validation_query_augmentations", "test_query_augmentations" : Can be true or false, defines if augmentations for query should be applied or not in each train/val/test phase.
- lr : Initial Learning rate
- loss: this section defines the added contrastive loss
- l_param is the coefficient of the added contrastive loss
- Choose to use one of two  or neither (cpl and angular). For cpl m_param is the negative samples per class and t_param is the temperature parameter (see paper). For angular angle is the alpha threshold and prototypes_as_anchors can be true or false.
- "num_epochs": the training epochs
- "multi_segm": True for BirdClef, voxceleb and FSD2018, false for the rest. If true, at the testing phase it applies majority vote over all segments of the same input.  
- "tie_strategy": Can be "" or "min_label" or "max_posterior". This handles ties at majority vote. "" picks the first label occured, "min_label" picks the minimum label (arithmetically), max_posterior picks the label with the highest posterior value.
- "relation_head": If true, RelationNets are used in stead of ProtoNets
- "n_training_tasks": The number of training tasks
- "n_testing_tasks": Number of test tasks
- "device": "cuda" or "cpu"
- "gpu_index": 0 , 1, 2 etc
- "scheduler_milestones": This is a list of scheduler milestones. When training epoch reaches each milestone, lr will be multiplied with scheduler_gamma
- "scheduler_gamma": as mentioned above
- "patience": for early stopping
- "experiment_folder": the experiment where the pt file will be saved
- "normalize_prototypes": if prototypes should be normalized when used for the contrastive loss calculation - can be true or false
- "project_prototypes": defines if prototypes should be passed through the projection head at the phase of the contrastive loss calculation. Projection head also normalizes them. 
- "specaug_params" : The parameters of spectrogram augmentations based on SpecAugment. If you want augmentations be sure to fill "use" = True. Applied only when "input_type" = "spec"
- waveaug_params" : The parameters of waveform agumentations. Applied only when "input_type" = "wav" and "use" = True

## 4. Scenarios

In this  section we want to show how to set the experiment_config.json file to run different scenarios. 
We  present only the parts of the json file which affect each scenario. The rest parameters of the config file should be included too.
### 1. Plain Prototypical Networks

```
"input_type": "spec"
"use_attention":false,
"use_contrastive":false,
"train_query_augmentations":false,
"validation_query_augmentations":false,
"test_query_augmentations":false,
"loss":
    {
        "l_param": a_number,
    
        "cpl":
        {
            "use": false,
            "m_param":a_number,
            "t_param":a_number
        },
        "angular":
        {
            "use":false,
            "angle": a_number,
            "prototypes_as_anchors": false
        }
    }
"relation_head": false

"specaug_params":
        {
        "use":false
        ...
}

```
### 2. Prototypical Networks with augmentations for batch enrichment.
```
In this setting every training input is augmented 3 times and used for few shot batch enrichement. Here we apply augmentations only at the training phase.

"input_type": "spec"
"use_attention":false,
"use_contrastive":false,
"train_query_augmentations":true,
"validation_query_augmentations":false,
"test_query_augmentations":false,
"loss":
    {
        "l_param": a_number,
    
        "cpl":
        {
            "use": false,
            "m_param":a_number,
            "t_param":a_number
        },
        "angular":
        {
            "use":false,
            "angle": a_number,
            "prototypes_as_anchors": false
        }
    }
"relation_head": false

"specaug_params":
        {
        "use":true,
        ....
}
```
### 3. Prototypical Networks with Augmentations and Self-attention
 In this scenario, all augmentations and the original input features are put in a list. The list is handled as a sequence by a self attention layer which outputs a concatenated vector of a unified representation. 
 ```


"input_type": "spec"
"use_attention":true,
"use_contrastive":false,
"train_query_augmentations":true,
"validation_query_augmentations":true,
"test_query_augmentations":true,
"loss":
    {
        "l_param": a_number,
    
        "cpl":
        {
            "use": false,
            "m_param":a_number,
            "t_param":a_number
        },
        "angular":
        {
            "use":false,
            "angle": a_number,
            "prototypes_as_anchors": false
        }
    }
"relation_head": false

"specaug_params":
        {
        "use":true,
        ....
}
```

### 4. Prototypical Networks + Contrastive learning
Here Prototypical Networks are combined with two versions of contrastive loss. The cpl loss and the angular loss. At this setting we do not have any augmentations.
Since we do not use the attention layer, the input dimension of the projection head should be the same as the output of the backbone. Change model_config.json accordingly (set projection_head['input_dim'] to 64 in stead of 256).
In the beloww example we use the cpl loss but you can use also the angular loss. Be sure to set cpl.use to false and angular.use to true for this.
 ```
"input_type": "spec"
"use_attention":false,
"use_contrastive":true,
"train_query_augmentations":false,
"validation_query_augmentations":false,
"test_query_augmentations":false,
"loss":
    {
        "l_param": a_number,
    
        "cpl":
        {
            "use": true,
            "m_param":a_number,
            "t_param":a_number
        },
        "angular":
        {
            "use":false,
            "angle": a_number,
            "prototypes_as_anchors": false
        }
    }
"relation_head": false

"specaug_params":
        {
        "use":false,
        ....
}
```

### 5. Prototypical Networks + Augmentations + Contrastive loss
In this example we use the angular loss and prototypes as anchors
```
"input_type": "spec"
"use_attention":true,
"use_contrastive":true,
"train_query_augmentations":true,
"validation_query_augmentations":true,
"test_query_augmentations":true,
"loss":
    {
        "l_param": a_number,
    
        "cpl":
        {
            "use": false,
            "m_param":a_number,
            "t_param":a_number
        },
        "angular":
        {
            "use":true,
            "angle": a_number,
            "prototypes_as_anchors": true
        }
    }
"relation_head": false

"specaug_params":
        {
        "use":true,
        ....
}
```

## 5. ModelConfig
This is a configuration file defining dynamically the internal architecture of each module:
```
{
  "CNN":
      {
    "in_channels":1,
    "hidden_channels":64,
    "pool_dim":[3, 3],
    "out_dim":64
      },
  
  "Hybrid":
      {
    "in_channels":1,
    "seq_layers":1,
    "seq_type":"RNN",
    "bidirectional":false,
    "hidden_channels":64,
    "pool_dim":[3, 3],
    "out_dim":64
      },
  
  "Attention":
      {
      "embed_dim":64,
      "num_heads":1,
      "ffn_dim":256,
      "dropout":0.1
      },
  
  "Projection":
      {
      "input_dim":256,
      "hidden_dim":128,
      "output_dim":256
      },
      
  "Relation":
      {
      "input_dim":512,
      "hidden_dim1":256,
      "hidden_dim2":128,
      "hidden_dim3":256,
      "out_dim":1
      }
    }
```




## 6. Hyperparameters
In this section we provide with a table with the best hyperparameters for augmentations, training, projection_heads, cpl and apl losses found by using optuna, a library for hyperparameter optimization.
| Dataset   | mask_param | W  | num_mask | p       | lr       | gamma   | Projection Head    |
|-----------|------------|----|----------|---------|----------|---------|---------------------|
| ESC-50    | 7          | 20 | 2        | 0.3127  | 0.00038  | 0.376   | 256-128-256         |
| FSD       | 16         | 22 | 1        | 0.282   | 0.0007   | 0.4482  | 256-512-256         |
| Nsynth    | 9          | 36 | 1        | 0.42157 | 0.0008   | 0.48426 | 256-64-64           |
| BirdClef  | 6          | 29 | 1        | 0.298   | 0.00149  | 0.59974 | 256-128-256         |
| VoxCeleb  | 6          | 29 | 1        | 0.298   | 0.00149  | 0.59974 | 256-128-256         |


For FS+CPL parameters we have : 

| Dataset   | l_param  | T_param | M     |
|-----------|----------|---------|-------|
| ESC-50    | 1.7235   | 6.0488  | 3     |
| FSD       | 2.022308 | 9.2361  | 5     |
| Nsynth    | 8.566    | 4.081   | 3     |
| BirdClef  | 9.1625   | 2.6981  | 4.0000|
| VoxCeleb  | 9.1625   | 2.6981  | 4.0000|


For FS+APL parameters we have: 

| Dataset   | angle | prototypes as anchors |
|-----------|-------|------------------------|
| ESC-50    | 15    | TRUE                   |
| FSD       | 30    | FALSE                  |
| Nsynth    | 15    | FALSE                  |
| BirdClef  | 15    | FALSE                  |
| VoxCeleb  | 0     | TRUE                   |




## 7. Run
To run an experiment you can use this line: 
```bash
cd audio-few-shot-learning
python3 src/train_test.py -e [path_to_experiment_config.json] -m [path_to_model_config.json]
```







