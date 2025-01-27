import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets.datasets import MetaAudioDataset
from utils.augmentations import SpecAugment, WaveAugment
import numpy as np

def augment_spectrogram(item,experiment_config):
    augmentation_module = SpecAugment(experiment_config)
    augmented_spec_list = augmentation_module.apply_augmentations(item)
    return augmented_spec_list
    
def augment_waveform(item,experiment_config):
    item = item.numpy()
    augmentation_module = WaveAugment(experiment_config)
    augmented_wav_list = augmentation_module.apply_augmentations(item)
    return augmented_wav_list

def sample_episode(dataset, n_classes , k_support, k_query , is_test, device, feat_extractor):
    multi_segm = dataset.multi_segm
    class_to_label = dataset.class_to_label
    class_labels = list(class_to_label.values())
    sampled_classes = sorted(random.sample(class_labels, n_classes))
    remapped_label_mapping = {original_label: new_label for new_label,original_label in enumerate(sampled_classes)}
    support_set = []
    support_labels = []
    query_set = []
    query_labels = []
    query_counter = 0 
    audio_ids = []
    support_list = []
    query_list = []  
        
    for class_label in sampled_classes:
        class_name = {v: k for k, v in class_to_label.items()}[class_label]
        class_indices = dataset.data_df[dataset.data_df['label'] == class_name]['index_column'].tolist()
        random.shuffle(class_indices)

    # Ensure enough indices for support and query sets
        if len(class_indices) < k_support + k_query:
            raise ValueError(f"Not enough samples for class {class_name}. "
                            f"Available: {len(class_indices)}, required: {k_support + k_query}")

    # Split indices into support and query sets
        support_indices = class_indices[:k_support]
        query_indices = class_indices[k_support:k_support + k_query]

        if dataset.input_type == 'spec':

            ### SPECTROGRAMS!!!!
            for idx in support_indices:
                spectrogram, label = dataset[idx]  
                if spectrogram.shape[0] !=1:
                    random_pick = random.randint(0,spectrogram.shape[0] - 1)
                    spectrogram = spectrogram[random_pick].unsqueeze(0)
                support_set.append(spectrogram)
                support_labels.append(remapped_label_mapping[class_label])
                         
            #Create query set for this class
            for idx in query_indices:
                spectrogram, label = dataset[idx]  # Use __getitem__, original shape preserved
                if is_test == False:
                    if spectrogram.shape[0] !=1:
                        random_pick = random.randint(0,spectrogram.shape[0]-1)
                        spectrogram = spectrogram[random_pick].unsqueeze(0)
                query_set.append(spectrogram)
                query_labels.extend([remapped_label_mapping[class_label]]*spectrogram.shape[0])
                audio_ids.extend([query_counter]*spectrogram.shape[0])
                    
                query_counter = query_counter + 1
        ## WAVS
        else:
            for idx in support_indices:
                wav, label = dataset[idx]  
                if multi_segm == True:
                    wav_segment_list = variable_wav_splits(wav)
                    random_pick = random.randint(0,len(wav_segment_list)-1)
                    wav_picked = wav_segment_list[random_pick]
                    wav_picked = torch.from_numpy(wav_picked).reshape(1,-1)
                        
                else:
                    wav_picked = torch.from_numpy(wav).reshape(1,-1)
                support_set.append(wav_picked)
                support_labels.append(remapped_label_mapping[class_label])

            for idx in query_indices:
                wav, label = dataset[idx]
                if multi_segm == True:
                    wav_segment_list = variable_wav_splits(wav)
                    if is_test == False:
                        random_pick = random.randint(0,len(wav_segment_list)-1)
                        wav_picked = wav_segment_list[random_pick]
                        wav_picked = torch.from_numpy(wav_picked.reshape(1,-1))
                                                                    
                    else:
                        torch_list = [torch.from_numpy(wav.reshape(1,-1)) for wav in wav_segment_list]
                        wav_picked = torch.cat(torch_list, dim = 0)
                else:
                    wav_picked = torch.from_numpy(wav.reshape(1,-1))
                               
                query_set.append(wav_picked)
                query_labels.extend([remapped_label_mapping[class_label]]*wav_picked.shape[0])
                audio_ids.extend([query_counter]*wav_picked.shape[0]) 
                query_counter = query_counter + 1
        
    support_set = torch.cat(support_set , dim = 0)
    query_set = torch.cat(query_set, dim = 0)

    if dataset.input_type == 'spec':
        if dataset.specaug_use == True:
            support_set_list = augment_spectrogram(item = support_set, experiment_config = dataset.experiment_config)
            query_set_list = augment_spectrogram(item = query_set, experiment_config = dataset.experiment_config)

        else: 
            support_set_list = [support_set]
            query_set_list = [query_set]

    elif dataset.input_type == 'wav':
        support_set_list = augment_waveform(item = support_set, experiment_config= dataset.experiment_config)
        query_set_list = augment_waveform(item = query_set,experiment_config = dataset.experiment_config)
    
    support_tensor = torch.cat(support_set_list)
    query_tensor = torch.cat(query_set_list)


    support_batch_length = n_classes*k_support
    query_batch_length = n_classes * k_query
    if dataset.input_type == 'wav':
        mean,std = dataset.get_normalization_stats()
        stacked_waveforms = torch.cat([support_tensor,query_tensor], dim =0).to(device)
        stacked_spectrograms =  mel_spec_function_gpu(stacked_waveforms,mel_transform=feat_extractor)
        norm_stacked_spectrograms  = (stacked_spectrograms - mean) / std
        norm_stacked_spectrograms = norm_stacked_spectrograms.unsqueeze(1)
    else:
        norm_stacked_spectrograms = torch.cat([support_tensor,query_tensor], dim = 0).to(device)
    start= 0
    if is_test == False:
        for i in range(len(support_set_list)):
            support = norm_stacked_spectrograms[start:start+support_batch_length,]
            support_list.append(support)
            start = start + support_batch_length
            
        for i in range(len(query_set_list)):
            query = norm_stacked_spectrograms[start:start+query_batch_length]
            start = start + query_batch_length
            query_list.append(query)

    elif is_test == True:
        for i in range(len(support_set_list)):
            support = norm_stacked_spectrograms[start:start+support_batch_length]
            start = start+support_batch_length
            support_list.append(support)
        for i in range(len(query_set_list)):
            query = norm_stacked_spectrograms[start:start+len(audio_ids)]
            start = start + len(audio_ids)
            query_list.append(query)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)
    audio_ids = torch.tensor(audio_ids)
    return support_list, support_labels, query_list, query_labels, audio_ids 


def variable_wav_splits(sample):
    #sample = torch.from_numpy(sample)
    length_s = 5
    expected_size = length_s * 16000
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
    return raw_splits

def mel_spec_function_gpu(x,mel_transform):
    mel_spec = mel_transform(x)
    epsilon = torch.finfo(mel_spec.dtype).eps
    return 20.0 / 2 * torch.log10(mel_spec + epsilon)



if __name__ == "__main__":
    import json
    from datasets.datasets import MetaAudioDataset
    import torchaudio
    with open("experiment_config.json", "r") as f:
        experiment_config = json.load(f)
    datset = MetaAudioDataset(experiment_config = experiment_config,root='/data/FSD2018', split='train')
    n_classes = 5
    k_support = 5
    k_query = 5
    device = 'cuda:1'
    feat_extractor  =  torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        power=2.0,
    ).to(device)
    support_list, support_labels, query_list, query_labels, audio_ids = sample_episode(datset,5,5,5 ,is_test=  True , device = device , feat_extractor=feat_extractor)
    for i in support_list:
        print(i.shape)
    print(support_labels)
    for i in query_list:
        print(i.shape)
    print(len(query_labels))
    print(len(audio_ids))