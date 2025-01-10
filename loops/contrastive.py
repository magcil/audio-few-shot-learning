import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)

from statistics import mean
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch
from loops.prototypical import evaluate
from callbacks.early_stopping import EarlyStopping
from datasets.task_sampler import generate_support_and_query
import torch.nn.functional as F
import numpy as np
from collections import Counter


def training_epoch(model, data_loader: DataLoader, optimizer: Optimizer, device, fsl_loss_fn, cpl_loss_fn, l_param,
                   project_prototypes, normalize_prototypes):
    all_loss = []
    model.train()
    fsl_loss_list = []
    cpl_loss_list = []
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
        ) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(support_images.to(device), support_labels.to(device))
            query_features = model(query_images.to(device))
            if model.relation_head:
                one_hot_support_labels =  F.one_hot(support_labels,  num_classes = torch.unique(support_labels).numel()).float()
                one_hot_support_labels = one_hot_support_labels.to(device)
                one_hot_query_labels = F.one_hot(query_labels,  num_classes = torch.unique(query_labels).numel()).float()
                one_hot_query_labels = one_hot_query_labels.to(device)
                relation_scores = query_features
                fsl_loss_fn =torch.nn.MSELoss()
                fsl_loss = fsl_loss_fn(relation_scores, one_hot_query_labels)
            else:
                fsl_loss = fsl_loss_fn(model.prototypes, query_features, query_labels.to(device))

            cpl_query_features, prototypes = model.contrastive_forward(project_prototypes)
            if project_prototypes == True:
                normalize_prototypes = False
            if normalize_prototypes == True:
                prototypes = F.normalize(prototypes, p=2.0, dim=1, eps=1e-12, out=None)
            cpl_loss = cpl_loss_fn(prototypes, cpl_query_features, query_labels.to(device))
            final_loss = fsl_loss + l_param * cpl_loss
            final_loss.backward()
            optimizer.step()
            all_loss.append(final_loss.item())
            fsl_loss_list.append(fsl_loss.item())
            cpl_loss_list.append(cpl_loss.item())
            tqdm_train.set_postfix(loss=mean(all_loss), loss_fsl=mean(fsl_loss_list), loss_cpl=mean(cpl_loss_list))

    return {"loss": mean(all_loss), "fsl_loss": mean(fsl_loss_list), "cpl_loss": mean(cpl_loss_list)}


def contrastive_training_loop(model, training_loader, validation_loader, optimizer, device, fsl_loss_fn, cpl_loss_fn,
                              l_param, epochs, train_scheduler, patience, results_path, project_prototypes,
                              normalize_prototypes):

    ear_stopping = EarlyStopping(path=os.path.join(PROJECT_PATH, "experiments", results_path, "model.pt"),
                                 patience=patience,
                                 verbose=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch:03}/{epochs+1:03}")
        average_training_loss, average_fsl_loss, average_cpl_loss = training_epoch(
            model=model,
            data_loader=training_loader,
            optimizer=optimizer,
            device=device,
            fsl_loss_fn=fsl_loss_fn,
            cpl_loss_fn=cpl_loss_fn,
            l_param=l_param,
            project_prototypes=project_prototypes,
            normalize_prototypes=normalize_prototypes)

        validation_accuracy, validation_accuracy_std = evaluate(model=model,
                                                                data_loader=validation_loader,
                                                                device=device)
        ear_stopping(val_accuracy=validation_accuracy, model=model, epoch=epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break

        train_scheduler.step()
    saved_model_pt_path = os.path.join(PROJECT_PATH, "experiments", results_path, "model.pt")
    model.load_state_dict(torch.load(saved_model_pt_path))
    trained_model = model

    return trained_model


def contrastive_testing_loop(trained_model, testing_loader, device):
    test_accuracy, accuracy_std = evaluate(model=trained_model, data_loader=testing_loader, device=device)
    return {"test_accuracy": test_accuracy, "test_accuracy_std": accuracy_std}


def calculate_majority_vote_accuracy(predicted_labels,
                                     spectrogram_ids,
                                     query_labels,
                                     posterior_values,
                                     tie_strategy="min_label"):
    """
    Calculate accuracy using majority voting with optional tie-breaking strategies.

    Args:
        predicted_labels: Tensor or array of predicted labels, shape [total_segments].
        spectrogram_ids: Tensor or array of segment IDs, shape [total_segments].
        query_labels: Tensor or array of true labels, shape [total_segments].
        posterior_values: Tensor or array of posterior values, shape [total_segments].
        tie_strategy: Tie-breaking strategy. "min_label" or "max_posterior".

    Returns:
        accuracy: Float, the accuracy based on majority vote.
    """
    # Ensure tensors are on the CPU and convert to numpy arrays
    predicted_labels = predicted_labels.cpu().numpy() if isinstance(predicted_labels,
                                                                    torch.Tensor) else predicted_labels
    spectrogram_ids = spectrogram_ids.cpu().numpy() if isinstance(spectrogram_ids, torch.Tensor) else spectrogram_ids
    query_labels = query_labels.cpu().numpy() if isinstance(query_labels, torch.Tensor) else query_labels
    posterior_values = posterior_values.cpu().numpy() if isinstance(posterior_values,
                                                                    torch.Tensor) else posterior_values

    # Get unique segment IDs
    unique_segments = np.unique(spectrogram_ids)

    # Initialize counters for correct segments
    correct_segments = 0
    total_segments = len(unique_segments)

    for segment in unique_segments:
        # Get indices of the current segment
        indices = [i for i, seg_id in enumerate(spectrogram_ids) if seg_id == segment]

        # Extract predictions, true labels, and posterior values for this segment
        segment_predictions = [int(predicted_labels[i]) for i in indices]
        segment_true_labels = [int(query_labels[i]) for i in indices]
        segment_posteriors = [posterior_values[i] for i in indices]

        # Majority vote for predictions
        counts = Counter(segment_predictions)
        max_count = max(counts.values())  # Get the maximum frequency

        # Get all labels tied for the maximum count
        tied_labels = [label for label, count in counts.items() if count == max_count]

        if len(tied_labels) == 1:
            # No tie, select the label with the highest count
            majority_prediction = tied_labels[0]
        else:
            if tie_strategy == "min_label":
                # Tie-breaking by selecting the smallest label
                majority_prediction = min(tied_labels)
            elif tie_strategy == "max_posterior":
                # Tie-breaking by selecting the label with the highest posterior value
                max_posterior = -np.inf
                majority_prediction = None
                for i, label in enumerate(segment_predictions):
                    if label in tied_labels and segment_posteriors[i] > max_posterior:
                        max_posterior = segment_posteriors[i]
                        majority_prediction = label
            else:
                majority_prediction = tied_labels[0]

        # Debugging output to verify logic

        # Check the true label consistency (assumes it's the same within a segment)
        true_label = segment_true_labels[0]

        # Compare majority prediction to true label
        if majority_prediction == true_label:
            correct_segments += 1

    # Calculate accuracy
    accuracy = correct_segments / total_segments
    return accuracy


def multisegment_testing_loop(test_dataset, n_classes, k_support, k_query, num_test_tasks, trained_model, device,
                              tie_strategy):
    list_of_accuracies = []
    for i in range(num_test_tasks):
        ## Generate a test episode:
        support_set, support_labels, query_set, query_labels, spectrogram_ids = generate_support_and_query(
            dataset=test_dataset, n_classes=n_classes, k_support=k_support, k_query=k_query)
        support_set = support_set.to(device)
        support_labels = support_labels.to(device)
        query_set = query_set.to(device)
        query_labels = query_labels.to(device)
        spectrogram_ids = spectrogram_ids.to(device)
        trained_model.process_support_set(support_set, support_labels)
        with torch.no_grad():
            predictions = trained_model(query_set, inference=True)
            predicted_labels = torch.max(predictions, 1)[1]
            posterior_values = torch.max(predictions, 1)[0]
            task_accuracy = calculate_majority_vote_accuracy(predicted_labels=predicted_labels,
                                                             spectrogram_ids=spectrogram_ids,
                                                             query_labels=query_labels,
                                                             tie_strategy=tie_strategy,
                                                             posterior_values=posterior_values)
            list_of_accuracies.append(task_accuracy)
    mean_accuracy = np.mean(list_of_accuracies)
    std_accuracy = np.std(list_of_accuracies)

    msg = {"mean_accuracy": mean_accuracy, "accuracy_std": std_accuracy}
    return msg
