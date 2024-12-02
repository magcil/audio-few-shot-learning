import random
from typing import Dict, Iterator, List, Tuple, Union
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import Tensor
from torch.utils.data import Sampler
from datasets.few_shot_dataset import FewShotDataset

GENERIC_TYPING_ERROR_MESSAGE = ("Check out the output's type of your dataset's __getitem__() method."
                                "It must be a Tuple[Tensor, int] or Tuple[Tensor, 0-dim Tensor].")


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: FewShotDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int
    
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have implement get_labels() from
                FewShotDataset.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label: Dict[int, List[int]] = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

        self._check_dataset_size_fits_sampler_parameters()

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        """
        Sample n_way labels uniformly at random,
        and then sample n_shot + n_query items for each label, also uniformly at random.
        Yields:
            a list of indices of length (n_way * (n_shot + n_query))
        """
        for _ in range(self.n_tasks):
            yield torch.cat([
                torch.tensor(random.sample(self.items_per_label[label], self.n_shot + self.n_query))
                for label in random.sample(sorted(self.items_per_label.keys()), self.n_way)
            ]).tolist()

    def episodic_collate_fn(
            self, input_data: List[Tuple[Tensor, Union[Tensor,
                                                       int]]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images of shape (n_way * n_shot, n_channels, height, width),
                - their labels of shape (n_way * n_shot),
                - query images of shape (n_way * n_query, n_channels, height, width)
                - their labels of shape (n_way * n_query),
                - the dataset class ids of the class sampled in the episode
        """
        input_data_with_int_labels = self._cast_input_data_to_tensor_int_tuple(input_data)
        true_class_ids = list({x[1] for x in input_data_with_int_labels})
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data_with_int_labels])
        all_images = all_images.reshape((self.n_way, self.n_shot + self.n_query, *all_images.shape[1:]))
        all_labels = torch.tensor([true_class_ids.index(x[1]) for x in input_data_with_int_labels]).reshape(
            (self.n_way, self.n_shot + self.n_query))
        support_images = all_images[:, :self.n_shot].reshape((-1, *all_images.shape[2:]))
        query_images = all_images[:, self.n_shot:].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, :self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot:].flatten()
        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )

    @staticmethod
    def _cast_input_data_to_tensor_int_tuple(
            input_data: List[Tuple[Tensor, Union[Tensor, int]]]) -> List[Tuple[Tensor, int]]:
        """
        Check the type of the input for the episodic_collate_fn method, and cast it to the right type if possible.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            the input data with the labels cast to int
        Raises:
            TypeError : Wrong type of input images or labels
            ValueError: Input label is not a 0-dim tensor
        """
        for image, label in input_data:
            if not isinstance(image, Tensor):
                raise TypeError(f"Illegal type of input instance: {type(image)}. " + GENERIC_TYPING_ERROR_MESSAGE)
            if not isinstance(label, int):
                if not isinstance(label, Tensor):
                    raise TypeError(f"Illegal type of input label: {type(label)}. " + GENERIC_TYPING_ERROR_MESSAGE)
                if label.dtype not in {
                        torch.uint8,
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                }:
                    raise TypeError(f"Illegal dtype of input label tensor: {label.dtype}. " +
                                    GENERIC_TYPING_ERROR_MESSAGE)
                if label.ndim != 0:
                    raise ValueError(f"Illegal shape for input label tensor: {label.shape}. " +
                                     GENERIC_TYPING_ERROR_MESSAGE)

        return [(image, int(label)) for (image, label) in input_data]

    def _check_dataset_size_fits_sampler_parameters(self):
        """
        Check that the dataset size is compatible with the sampler parameters
        """
        self._check_dataset_has_enough_labels()
        self._check_dataset_has_enough_items_per_label()

    def _check_dataset_has_enough_labels(self):
        if self.n_way > len(self.items_per_label):
            raise ValueError(f"The number of labels in the dataset ({len(self.items_per_label)} "
                             f"must be greater or equal to n_way ({self.n_way}).")

    def _check_dataset_has_enough_items_per_label(self):
        number_of_samples_per_label = [len(items_for_label) for items_for_label in self.items_per_label.values()]
        minimum_number_of_samples_per_label = min(number_of_samples_per_label)
        label_with_minimum_number_of_samples = number_of_samples_per_label.index(minimum_number_of_samples_per_label)
        if self.n_shot + self.n_query > minimum_number_of_samples_per_label:
            raise ValueError(
                f"Label {label_with_minimum_number_of_samples} has only {minimum_number_of_samples_per_label} samples"
                f"but all classes must have at least n_shot + n_query ({self.n_shot + self.n_query}) samples.")


def sample_support_and_query(dataset, n_classes, k_support, k_query):
    """
    Sample support and query sets from the dataset using index_column for unique identifiers.
    
    Args:
        dataset: Dataset object with multi-segment enabled.
        n_classes: Number of classes to sample.
        k_support: Number of spectrograms per class for the support set.
        k_query: Number of spectrogram shots per class for the query set.
    
    Returns:
        support_set: List of support spectrograms for each class.
        query_set: List of query spectrograms for each class.
        sampled_classes: List of sampled class labels (mapped to 0, ..., n_classes-1).
    """
    # Map from class names to numeric labels
    class_to_label = dataset.class_to_label 
    # Get unique numeric class labels
    class_labels = list(class_to_label.values())
    # Randomly sample n classes and sort them
    sampled_classes = sorted(random.sample(class_labels, n_classes))  
    # Map sampled class labels to 0, ..., n_classes-1
    remapped_label_mapping = {original_label: new_label for new_label, original_label in enumerate(sampled_classes)}
    support_set = []
    query_set = []
    
    for class_label in sampled_classes:
        # Get the class name for this numeric label
        class_name = {v: k for k, v in class_to_label.items()}[class_label]
        
        # Get all spectrograms for this class using index_column
        class_indices = dataset.data_df[dataset.data_df['label'] == class_name]['index_column'].tolist()
        # Shuffle indices for random sampling
        random.shuffle(class_indices)
        
        # Ensure enough indices for support and query sets
        if len(class_indices) < k_support + k_query:
            raise ValueError(f"Not enough samples for class {class_name}. "
                             f"Available: {len(class_indices)}, required: {k_support + k_query}")
        
        # Split indices into support and query sets
        support_indices = class_indices[:k_support]
        query_indices = class_indices[k_support:k_support + k_query]
        
        # Create support set for this class
        support_spectrograms = []
        for idx in support_indices:
            spectrogram, label = dataset[idx]  # Use __getitem__ for support
            if spectrogram.shape[0] == 1:
                support_spectrograms.append(spectrogram.unsqueeze(1))  # Shape: [1,1, 128, 157]
            else:
                random_pick = random.randint(0, spectrogram.shape[0] - 1)
                chosen_spectrogram = spectrogram[random_pick].unsqueeze(0).unsqueeze(0)
                support_spectrograms.append(chosen_spectrogram)
        
        # Create query set for this class
        query_shots = []
        for idx in query_indices:
            spectrogram, label = dataset[idx]  # Use __getitem__, original shape preserved
            query_shots.append(spectrogram.unsqueeze(1))  # Shape: [num_specs, 128, 157]
        
        # Append to the final sets
        support_set.append(support_spectrograms)  # Shape: [k_support, 1, 128, 157]
        query_set.append(query_shots)  # Shape: [k_query, num_specs, 128, 157]
    # Replace sampled_classes with remapped labels
    sampled_classes = [remapped_label_mapping[class_label] for class_label in sampled_classes]  
    return support_set, query_set, sampled_classes

def prepare_stacked_support(support_set, sampled_classes, k_support):
    """
    Stack all support spectrograms into a single tensor and create corresponding labels.

    Args:
        support_set: List of support spectrograms for each class.
                     Shape: [n_classes, k_support, 1, 128, 157].
        sampled_classes: List of class labels corresponding to the support_set indices.
        k_support: Number of support shots per class.

    Returns:
        stacked_support: Stacked tensor of shape [k_support * n_classes, 1, 128, 157].
        support_labels: Corresponding labels of shape [k_support * n_classes].
    """
    stacked_support = []
    support_labels = []

    for class_idx, class_support in enumerate(support_set):
        # Flatten the support set for the class and add labels
        stacked_support.extend(class_support)  # Extend the list with the tensors
        support_labels.extend([sampled_classes[class_idx]] * k_support)  # Repeat the label k_support times

    # Stack all support spectrograms into a single tensor
    stacked_support = torch.cat(stacked_support, dim=0)  # Shape: [k_support * n_classes, 1, 128, 157]

    # Convert support_labels to a tensor
    support_labels = torch.tensor(support_labels)  # Shape: [k_support * n_classes]

    return stacked_support, support_labels

def prepare_stacked_query(query_set, sampled_classes):
    """
    Stack all query spectrograms into a single tensor and create corresponding labels.

    Args:
        query_set: List of query spectrograms for each class.
                   Shape: [n_classes, k_query, num_segments, 1, freq_bins, time_bins].
        sampled_classes: List of class labels corresponding to the query_set indices.
        k_query: Number of query shots per class.

    Returns:
        stacked_query: Stacked tensor of shape [total_num_segments, 1, freq_bins, time_bins].
        query_labels: Corresponding labels of shape [total_num_segments].
    """
    stacked_query = []
    query_labels = []

    for class_idx, class_queries in enumerate(query_set):
        for query_shot in class_queries:
            # query_shot is of shape [num_segments, 1, freq_bins, time_bins]
            num_segments = query_shot.shape[0]

            # Add all spectrograms in this query_shot to the stack
            stacked_query.append(query_shot)  # Append [num_segments, 1, freq_bins, time_bins]

            # Add corresponding labels for each segment in the shot
            query_labels.extend([sampled_classes[class_idx]] * num_segments)

    # Concatenate all spectrograms along dimension 0
    stacked_query = torch.cat(stacked_query, dim=0)  # Shape: [total_num_segments, 1, freq_bins, time_bins]

    # Convert query_labels to a tensor
    query_labels = torch.tensor(query_labels)  # Shape: [total_num_segments]

    return stacked_query, query_labels



def prepare_stacked_query1(query_set, sampled_classes):
    """
    Stack all query spectrograms into a single tensor and create corresponding labels.
    Additionally, track which query segments come from the same spectrogram.

    Args:
        query_set: List of query spectrograms for each class.
                   Shape: [n_classes, k_query, num_segments, 1, freq_bins, time_bins].
        sampled_classes: List of class labels corresponding to the query_set indices.

    Returns:
        stacked_query: Stacked tensor of shape [total_num_segments, 1, freq_bins, time_bins].
        query_labels: Corresponding labels of shape [total_num_segments].
        spectrogram_ids: List of spectrogram IDs indicating the origin of each segment.
                         Shape: [total_num_segments].
    """
    stacked_query = []
    query_labels = []
    spectrogram_ids = []  # To track the origin of each segment
    spectrogram_counter = 0  # Unique ID for each spectrogram

    for class_idx, class_queries in enumerate(query_set):
        for query_shot in class_queries:
            # query_shot is of shape [num_segments, 1, freq_bins, time_bins]
            num_segments = query_shot.shape[0]

            # Add all spectrograms in this query_shot to the stack
            stacked_query.append(query_shot)  # Append [num_segments, 1, freq_bins, time_bins]

            # Add corresponding labels for each segment in the shot
            query_labels.extend([sampled_classes[class_idx]] * num_segments)

            # Track which segments came from the same spectrogram
            spectrogram_ids.extend([spectrogram_counter] * num_segments)

            # Increment spectrogram ID for the next spectrogram
            spectrogram_counter += 1

    # Concatenate all spectrograms along dimension 0
    stacked_query = torch.cat(stacked_query, dim=0)  # Shape: [total_num_segments, 1, freq_bins, time_bins]

    # Convert query_labels and spectrogram_ids to tensors
    query_labels = torch.tensor(query_labels)  # Shape: [total_num_segments]
    spectrogram_ids = torch.tensor(spectrogram_ids)  # Shape: [total_num_segments]

    return stacked_query, query_labels, spectrogram_ids


def generate_support_and_query(dataset, n_classes, k_support, k_query):
    support_set, query_set, sampled_classes =  sample_support_and_query(dataset = dataset, n_classes = n_classes, k_support = k_support, k_query = k_query)
    stacked_support_set, support_labels = prepare_stacked_support(support_set = support_set, sampled_classes = sampled_classes, k_support=k_support)
    stacked_query_set, query_labels,spectrogram_ids = prepare_stacked_query1(query_set = query_set, sampled_classes = sampled_classes)

    return stacked_support_set, support_labels, stacked_query_set, query_labels,spectrogram_ids


