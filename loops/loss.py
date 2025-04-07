import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from pytorch_metric_learning.losses import AngularLoss  
from pytorch_metric_learning.miners import AngularMiner
import torch.nn as nn
import torch
import torch.nn.functional as F


class FSL_Loss(nn.Module):
    """
    Implementation of FewShot Classification as presented in
    https://arxiv.org/pdf/2101.09499
    """

    def __init__(self):
        super(FSL_Loss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nll_loss = nn.NLLLoss()

    def forward(self, prototypes: torch.Tensor, queries: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prototypes (torch.Tensor): 2D tensor containing the prototypes.
            queries (torch.Tensor): 2D tensor containing the queries.
            labels (torch.Tensor): 1D tensor with labels of queries.
        Returns:
            FewShot Classification Loss.
        """
        # D (N queries x N prototypes): Euclidean distances between each query and prototype
        D = (-1) * torch.cdist(x1=queries, x2=prototypes, p=2.0)

        log_softmax = self.log_softmax(D)
        return self.nll_loss(log_softmax, labels)

class AngularLossClass(nn.Module):
    def __init__(self,angle, prototypes_as_anchors):
        super(AngularLossClass, self).__init__()
        # Initialize the AngularLoss function
        self.loss_fn = AngularLoss()
        self.protoypes_as_anchors = prototypes_as_anchors
        self.angle = angle
        self.miner = AngularMiner(angle = self.angle)

    def forward(self, prototypes, queries, query_labels):
        """
        Compute the angular loss between prototypes and queries.

        Args:
            prototypes (torch.Tensor): Tensor of shape (num_prototypes, feature_dim),
                                       one prototype per class.
            queries (torch.Tensor): Tensor of shape (num_queries, feature_dim),
                                    with multiple queries per class.
            query_labels (torch.Tensor): Tensor of shape (num_queries,), indicating the class label
                                         for each query.

        Returns:
            torch.Tensor: The computed angular loss.
        """
        num_prototypes = prototypes.size(0)
        unique_labels = torch.unique(query_labels)
        assert num_prototypes == unique_labels.size(0)
        # Create labels for prototypes
        prototype_labels = torch.arange(num_prototypes)
        if self.protoypes_as_anchors:
            mined_examples = self.miner(
            embeddings=prototypes,   # Prototypes as anchors
            labels=prototype_labels, # Prototype labels
            ref_emb=queries,         # Queries as references (positives/negatives)
            ref_labels=query_labels)
            anchor_labels = torch.tensor([prototype_labels[i] for i in mined_examples[0]])
            embeddings = prototypes[mined_examples[0]]
            positive_query = queries[mined_examples[1]]
            negative_query = queries[mined_examples[2]]

            ref_emb = torch.cat([positive_query,negative_query])
            positive_labels = torch.tensor([query_labels[i] for i in mined_examples[1]])
            negative_labels = torch.tensor([query_labels[i] for i in mined_examples[2]])
            ref_labels = torch.cat([positive_labels, negative_labels])
            loss = self.loss_fn(embeddings, anchor_labels, ref_emb=ref_emb, ref_labels=ref_labels)
        else:
            # Create labels for prototypes
            prototype_labels = torch.arange(num_prototypes)
            prototype_labels = prototype_labels.to(prototypes.device)
            # Combine prototypes and queries into one tensor
            embeddings = torch.cat([prototypes, queries], dim=0)

            # Combine prototype labels and query labels
            labels = torch.cat([prototype_labels, query_labels], dim=0)
            miner_output = self.miner(embeddings, labels)

            # Compute the angular loss
            loss = self.loss_fn(embeddings, labels, miner_output)
        return loss

class CPL_Loss(nn.Module):
    """
    Implementation of Contrastive Prototype Learning Loss as 
    presented in https://arxiv.org/pdf/2101.0949.
    """

    def __init__(self, T: float = 1.0, M: int = 5):
        """
        Args:
            T (float): Temperature hyperparameter.
            M (int): Number of samples from each negative class.
        """
        super(CPL_Loss, self).__init__()
        self.T = T
        self.M = M
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nll_loss = nn.NLLLoss()
        self.device = None

    def forward(self, prototypes: torch.Tensor, queries: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prototypes (torch.Tensor): 2D tensor containing the prototypes.
            queries (torch.Tensor): 2D tensor containing the queries.
            labels (torch.Tensor): 1D tensor with labels of queries.
        Returns:
            Contrastive prototype loss.
        """

        # Tensor of shape N Queries x ((N Shot - 1) * M + 1)
        cos_sim, targets = self.similarity_sampling(prototypes, queries, labels, self.M)

        return (1 / queries.shape[0]) * self.nll_loss(self.log_softmax(cos_sim), targets)

    # Function to sample M samples from each of the other classes for each query vector
    def similarity_sampling(self, prototypes, queries, labels, M):
        unique_labels = labels.unique()  # Get unique class labels
        N_way = len(unique_labels)
        label_to_indices = {label.item(): torch.where(labels == label)[0] for label in unique_labels}
        self.device = prototypes.device

        # List to store exp cosine similarities
        cos_sim = []
        targets = []
        for i, (query, label) in enumerate(zip(queries, labels)):
            # Get samples from each of the other classes (not the current label's class)
            samples = []
            for other_label, indices in label_to_indices.items():
                if other_label != label.item():  # Skip the current query's class
                    # Sample M indices from the current class
                    selected_indices = indices[torch.randperm(len(indices))[:M]]
                    samples.append(queries[selected_indices])

            # Concatenate the samples from the N_Way other classes into a single tensor
            # Append the current query in the last row
            samples = torch.vstack([torch.cat(samples, dim=0), query.unsqueeze(0)])  # Shape: [(N_Way-1) * M + 1, D]
            # Update targets
            targets.append((N_way - 1) * M)
            # Get prototype
            corresponding_prototype = prototypes[label]
            # Calculate Cos sim
            cos_sim.append(F.cosine_similarity(x1=corresponding_prototype, x2=samples) / self.T)

        # Shape of N Queries x (N_Way - 1) * M + 1
        cos_sim = torch.stack(cos_sim)

        return cos_sim, torch.tensor(targets).to(self.device)


if __name__ == '__main__':
    prototypes = torch.rand(5,256)
    queries = torch.rand(25,256)
    query_labels = torch.tensor([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
    loss_fn = AngularLossClass()
    angular_loss = loss_fn(prototypes = prototypes, queries = queries, query_labels = query_labels)
    print(f"Angularloss:{angular_loss}")
    loss_fn = CPL_Loss()
    cpl_loss = loss_fn(prototypes = prototypes, queries = queries, labels = query_labels)
    print(f"CPL_LOSS::: {cpl_loss}")
