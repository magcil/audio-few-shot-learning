from abc import abstractmethod
from typing import List, Tuple
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset


class FewShotDataset(Dataset):
    """
    Abstract class for all datasets used in a context of Few-Shot Learning.
    The tools we use in few-shot learning, especially TaskSampler, expect an
    implementation of FewShotDataset.
    Compared to PyTorch's Dataset, FewShotDataset forces a method get_labels.
    This exposes the list of all items labels and therefore allows to sample
    items depending on their label.
    """

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        raise NotImplementedError("All PyTorch datasets, including few-shot datasets, need a __getitem__ method.")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("All PyTorch datasets, including few-shot datasets, need a __len__ method.")

    @abstractmethod
    def get_labels(self) -> List[int]:
        raise NotImplementedError("Implementations of FewShotDataset need a get_labels method.")


class WrapFewShotDataset(FewShotDataset):
    """
    Wrap a dataset in a FewShotDataset. This is useful if you have your own dataset
    and want to use it with the tools provided by EasyFSL such as TaskSampler.
    """

    def __init__(
        self,
        dataset: Dataset,
        image_position_in_get_item_output: int = 0,
        label_position_in_get_item_output: int = 1,
    ):
        """
        Wrap a dataset in a FewShotDataset.
        Args:
            dataset: dataset to wrap
            image_position_in_get_item_output: position of the image in the tuple returned
                by dataset.__getitem__(). Default: 0
            label_position_in_get_item_output: position of the label in the tuple returned
                by dataset.__getitem__(). Default: 1
        """
        if image_position_in_get_item_output == label_position_in_get_item_output:
            raise ValueError(
                "image_position_in_get_item_output and label_position_in_get_item_output must be different.")
        if (image_position_in_get_item_output < 0 or label_position_in_get_item_output < 0):
            raise ValueError(
                "image_position_in_get_item_output and label_position_in_get_item_output must be positive.")
        item_length = len(dataset[0])
        if (image_position_in_get_item_output >= item_length or label_position_in_get_item_output >= item_length):
            raise ValueError("Specified positions in output are out of range.")

        self.source_dataset = dataset
        self.labels = [
            source_dataset_instance[label_position_in_get_item_output]
            for source_dataset_instance in tqdm(dataset, desc="Scrolling dataset's labels...")
        ]
        self.image_position_in_get_item_output = image_position_in_get_item_output
        self.label_position_in_get_item_output = label_position_in_get_item_output

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        return (
            self.source_dataset[item][self.image_position_in_get_item_output],
            self.source_dataset[item][self.label_position_in_get_item_output],
        )

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> List[int]:
        return self.labels
