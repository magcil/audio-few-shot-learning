import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from pytorch_metric_learning.losses import AngularLoss  
from pytorch_metric_learning.miners import AngularMiner
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
from pytorch_metric_learning import losses
from typing import Tuple


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

class FSL_Loss_all_support(nn.Module):
    """
    FSL loss but instead of prototypes, 
    all support set samples are considered
    Implementation of NCA Loss
    """

    def __init__(self):
        super(FSL_Loss_all_support, self).__init__()
        self.nca_loss = losses.NCALoss(softmax_scale=1)

    def forward(self, support_set: torch.Tensor, queries: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = torch.cat((queries,support_set), dim=0)
        #print(self.nca_loss(embeddings,labels))
        return self.nca_loss(embeddings,labels)
    
    def forward_to_be(self, support_set: torch.Tensor, queries: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_set (torch.Tensor): 2D tensor containng episode support set for alternative loss calculation
            queries (torch.Tensor): 2D tensor containing the queries.
            labels (torch.Tensor): 1D tensor with labels of queries and support sets.
        Returns:
            FewShot Classification Loss.
        """
        embeddings = torch.cat((queries,support_set), dim=0) #.to(device)
        batch_size = embeddings.shape[0]
        # Compute pairwise Euclidean distances
        distance_matrix = torch.cdist(embeddings, embeddings, p=2) ** 2  # Squared Euclidean distance

        # Compute similarity scores (negative distances)
        similarity = -distance_matrix  # We take the negative because we want closer pairs to have higher scores

        loss = 0.0
        for i in range(batch_size):
            # Mask for same-class samples (excluding self)
            mask = (labels == labels[i]) & (torch.arange(batch_size) != i)
            
            # Get the probabilities of selecting each neighbor
            exp_sim = torch.exp(similarity[i])  # Exponential of similarity scores
            prob = exp_sim / (exp_sim.sum() - exp_sim[i])  # Exclude self
            
            # Compute loss for sample i
            loss -= torch.log(prob[mask].sum() + 1e-8)  # Avoid log(0)

        #print(loss / batch_size)
        return loss / batch_size  # Normalize over batch

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
    
class AngularLossClass_support(nn.Module):
    def __init__(self,angle,prototypes_as_anchors):
        super(AngularLossClass_support, self).__init__()
        # Initialize the AngularLoss function
        self.loss_fn = AngularLoss()
        self.protoypes_as_anchors = prototypes_as_anchors
        self.angle = angle
        self.miner = AngularMiner(angle = self.angle)

    def forward(self, support_set, queries, labels):
        """
        Compute the angular loss between prototypes and queries.

        Args:
            support_set (torch.Tensor): Tensor of shape (num_of_support_samples, feature_dim),
                                       for alternative loss calculation.
            queries (torch.Tensor): Tensor of shape (num_queries, feature_dim),
                                    with multiple queries per class.
            labels (torch.Tensor): Tensor of shape (num_queries+num_of_support,), indicating the class label
                                         for each query and support sample.

        Returns:
            torch.Tensor: The computed angular loss.
        """
        #num_prototypes = support_set.size(0)
        support_labels = labels[1]
        query_labels = labels[0]
        print(support_labels.shape)
        print(query_labels.shape)
        unique_labels = torch.unique(query_labels)
        assert torch.unique(support_labels).size(0) == unique_labels.size(0)
        # Create labels for prototypes
        #prototype_labels = torch.arange(num_prototypes)
        # TODO: use support as anchors
        if self.protoypes_as_anchors:
            print(support_set.shape)
            print("BEFORE MINER")
            mined_examples = self.miner(
            embeddings=support_set,   # Prototypes as anchors
            labels=support_labels, # Prototype labels
            ref_emb=queries,         # Queries as references (positives/negatives)
            ref_labels=query_labels)
            print("AFTER MINER")
            anchor_labels = torch.tensor([support_labels[i] for i in mined_examples[0]])
            embeddings = support_set[mined_examples[0]]
            print(len(embeddings))
            positive_query = queries[mined_examples[1]]
            negative_query = queries[mined_examples[2]]

            ref_emb = torch.cat([positive_query,negative_query])
            positive_labels = torch.tensor([query_labels[i] for i in mined_examples[1]])
            negative_labels = torch.tensor([query_labels[i] for i in mined_examples[2]])
            ref_labels = torch.cat([positive_labels, negative_labels])
            print("BEFORE LOSS")
            loss = self.loss_fn(embeddings, anchor_labels, ref_emb=ref_emb, ref_labels=ref_labels)
            print("AFTER LOSS")
        else:
            # Create labels for prototypes
            
            #prototype_labels = torch.arange(num_prototypes)
            #prototype_labels = prototype_labels.to(prototypes.device)
            
            # Combine prototypes and queries into one tensor
            embeddings = torch.cat([support_set, queries], dim=0)

            # Combine prototype labels and query labels
            labels = torch.cat([support_labels, query_labels], dim=0)
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

class CPL_Loss_all_support(nn.Module):
    """
    Implementation of Contrastive Prototype Learning Loss
    using all support set samples instaed of prototypes.
    """

    def __init__(self, T: float = 1.0, M: int = 5):
        """
        Args:
            T (float): Temperature hyperparameter.
            M (int): Number of samples from each negative class.
        """
        super(CPL_Loss_all_support, self).__init__()
        self.T = T
        self.M = M
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nll_loss = nn.NLLLoss()
        self.device = None

    def forward(self, support_set: torch.Tensor, queries: torch.Tensor, labels: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            support_set (torch.Tensor): 2D tensor containing the support set for alternative loss calculation.
            queries (torch.Tensor): 2D tensor containing the queries.
            labels Tuple[torch.Tensor, torch.Tensor]: tuple containing 2 1D tensors with labels of queries and support set.
        Returns:
            Contrastive prototype loss.
        """

        # Tensor of shape N Queries x ((N Shot - 1) * M + 1)
        cos_sim, targets = self.similarity_sampling(support_set, queries, labels, self.M)

        #print((1 / queries.shape[0]) * self.nll_loss(self.log_softmax(cos_sim), targets))
        return (1 / queries.shape[0]) * self.nll_loss(self.log_softmax(cos_sim), targets)

    # Function to sample M samples from each of the other classes for each query vector
    def similarity_sampling(self, support_set, queries, labels, M):
        support_set = [(sample, label) for sample, label in zip(support_set,labels[1])]
        labels = labels[0]
        unique_labels = labels.unique()  # Get unique class labels
        N_way = len(unique_labels)
        label_to_indices = {label.item(): torch.where(labels == label)[0] for label in unique_labels}
        self.device = support_set[0][0].device

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
            
            # Compute cosine similarities using all support samples of the same label instaed of using only class prototype
            for support_sample, support_label in support_set:
                if support_label == label.item():
                    # Calculate Cos sim
                    cos_sim.append(F.cosine_similarity(x1=support_sample, x2=samples) / self.T)
                    targets.append((N_way - 1) * M)
                else:
                    continue

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
