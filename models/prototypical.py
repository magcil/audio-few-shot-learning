"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""
from torch import Tensor
import os
import sys
import random
import torch
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.few_shot_classifier import FewShotClassifier
import torch.nn.functional as F


class PrototypicalNetworks(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def forward(
        self,
        query_images: Tensor,
        inference : Optional[bool] = True,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.
        """
        # Extract the features of query images
        query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)

        # Compute the euclidean distance from queries to prototypes
        scores = self.l2_distance_to_prototypes(query_features)

        return self.softmax_if_specified(scores)


class ContrastivePrototypicalNetworks(FewShotClassifier):

    def __init__(self, backbone, attention_model, projection_head, *args, **kwargs):
        # Call the parent class constructor and pass the remaining arguments
        super(ContrastivePrototypicalNetworks, self).__init__(*args, **kwargs)
        self.backbone = backbone
        self.attention_model = attention_model
        self.projection_head = projection_head

    def compute_features(self, images: Tensor) -> Tensor:
        original_feature_list = self.backbone(images)
        features = torch.stack(original_feature_list, dim=1)
        features = self.attention_model(features)
        return features

    def compute_query_features(self, images: Tensor) -> Tensor:
        self.query_feature_list = self.backbone(images)

        return self.query_feature_list

    def shuffle_augmentations(self, feature_list):
        augmentations = feature_list[1:]
        random.shuffle(augmentations)
        shuffled_features = torch.stack([feature_list[0]] + augmentations, dim=1)
        return shuffled_features

    def forward(self, query_images, inference=False, training_prototypes = None) :
        self.query_feature_list = self.compute_query_features(query_images)
        query_features = torch.stack(self.query_feature_list, dim=1)
        query_features = self.attention_model(query_features)
        self._raise_error_if_features_are_multi_dimensional(query_features)
        if inference == True:
            if training_prototypes:
                self.prototypes  = self.calibrate_prototypes(training_prototypes = training_prototypes, calibration_t = 15, calibration_a = 0.7)
            query_features = self.l2_distance_to_prototypes(query_features)

        return query_features

    def contrastive_forward(self, project_prototypes):
        shuffled_features = self.shuffle_augmentations(self.query_feature_list)
        shuffled_features = self.attention_model(shuffled_features)
        projected_features = self.projection_head(shuffled_features)
        if project_prototypes == True:
            projected_prototypes = self.projection_head(self.prototypes)
        else:
            projected_prototypes = self.prototypes
        return projected_features, projected_prototypes

    @staticmethod
    def is_transductive() -> bool:
        return False
    
    def calibrate_prototypes(self, training_prototypes,calibration_t,calibration_a):
        self.training_prototypes = training_prototypes
        batch_prototypes = self.prototypes
        ## Unpack Them
        mean_train_prototypes = []
        for label in sorted(training_prototypes.keys()):
            mean_train_prototype = torch.mean(training_prototypes[label], dim = 0)
            mean_train_prototypes.append(mean_train_prototype.unsqueeze(0))
        mean_train_prototypes = torch.concat(mean_train_prototypes, dim = 0)
        similarity_scores = F.cosine_similarity(mean_train_prototypes.unsqueeze(0), batch_prototypes.unsqueeze(1), dim = 2) * calibration_t
        weights = F.softmax(similarity_scores,dim = 1)
        calibration_delta = torch.matmul(weights,mean_train_prototypes)
        calibrated_prototypes = calibration_a * batch_prototypes + (1-calibration_a)*calibration_delta
        self.prototypes = calibrated_prototypes

        return self.prototypes


    



class ContrastivePrototypicalNetworksWithoutAttention(FewShotClassifier):
    def __init__(self, backbone, projection_head, *args, **kwargs):
        # Call the parent class constructor and pass the remaining arguments
        super(ContrastivePrototypicalNetworksWithoutAttention, self).__init__(*args, **kwargs)
        self.backbone = backbone
        self.projection_head = projection_head

    def compute_features(self, images: Tensor) -> Tensor:
        original_feature_list = self.backbone(images)
        features = torch.concat(original_feature_list, dim=0)
        return features
    
    def forward(self, query_images, inference=False):
        self.query_feature_list = self.compute_features(query_images)
        query_features = self.query_feature_list
        self._raise_error_if_features_are_multi_dimensional(query_features)
        if inference == True:
            query_features = self.l2_distance_to_prototypes(query_features)
        return query_features

    def contrastive_forward(self, project_prototypes):
        projected_features = self.projection_head(self.query_feature_list)
        if project_prototypes == True:
            projected_prototypes = self.projection_head(self.prototypes)
        else:
            projected_prototypes = self.prototypes
        return projected_features, projected_prototypes

    @staticmethod
    def is_transductive() -> bool:
        return False
