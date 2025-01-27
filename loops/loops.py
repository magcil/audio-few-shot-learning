import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)

from statistics import mean
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.optim import Optimizer
import torch
from callbacks.early_stopping import EarlyStopping
from datasets.batch_creation import sample_episode
import torch.nn.functional as F
import numpy as np
from collections import Counter


def training_epoch(model, dataset: Dataset, optimizer: Optimizer, num_train_tasks, device, fsl_loss_fn, cpl_loss_fn, l_param,
                   project_prototypes, normalize_prototypes, n_classes, k_support, k_query, feat_extractor):

    all_loss = []
    model.train()
    fsl_loss_list = []
    cpl_loss_list = []
    for task in tqdm(range(num_train_tasks), desc = "Training"):
            support_list, support_labels, query_list, query_labels, audio_ids = sample_episode(dataset = dataset,
                                                                                                           n_classes = n_classes, 
                                                                                                           k_support = k_support, 
                                                                                                           k_query = k_query, 
                                                                                                                   is_test = False, device = device ,feat_extractor= feat_extractor)


            optimizer.zero_grad()
            model.process_support_set(support_list, support_labels.to(device))
            query_features = model(query_list)
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

    return {"loss": mean(all_loss), "fsl_loss": mean(fsl_loss_list), "cpl_loss": mean(cpl_loss_list)}

def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor):
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    model.process_support_set(support_images, support_labels)
    with torch.no_grad():
        predictions = model(query_images, inference=True)
    number_of_correct_predictions = ((torch.max(predictions, 1)[1] == query_labels).sum().item())

    return number_of_correct_predictions, len(query_labels)


def evaluate_single_segment(model, dataset, num_val_tasks, device, n_classes, k_support, k_query, feat_extractor):
    # List to store accuracies for each task
 

    accuracies = []

    model.eval()
    with torch.no_grad():
        for task in tqdm(range(num_val_tasks), desc = "Validation"):
                support_list, support_labels, query_list, query_labels, _ = sample_episode(dataset = dataset,
                                                                                                           n_classes = n_classes, 
                                                                                                           k_support = k_support, 
                                                                                                           k_query = k_query, 
                                                                                                           is_test = False, device = device , feat_extractor=feat_extractor)
                support_list = [tensor.to(device) for tensor in support_list]
                query_list = [tensor.to(device)for tensor in query_list]
                correct, total = evaluate_on_one_task(
                    model,
                    support_list,
                    support_labels.to(device),
                    query_list,
                    query_labels.to(device),
                )

                # Calculate accuracy for this task and store it
                task_accuracy = correct / total
                accuracies.append(task_accuracy)

    # Calculate mean and standard deviation of accuracies
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return mean_accuracy, std_accuracy


def contrastive_training_loop(model, train_dataset, validation_dataset, optimizer,num_train_tasks,num_val_tasks, device, fsl_loss_fn, cpl_loss_fn,
                              l_param, epochs, train_scheduler, patience, results_path, project_prototypes,
                              normalize_prototypes,n_classes, k_support, k_query, feat_extractor):
    
    ear_stopping = EarlyStopping(path=os.path.join(PROJECT_PATH, "experiments", results_path, "model.pt"),
                                 patience=patience,
                                 verbose=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch:03}/{epochs+1:03}")
        average_training_loss, average_fsl_loss, average_cpl_loss = training_epoch(model = model, 
                                                                                   dataset = train_dataset, 
                                                                                   optimizer = optimizer, 
                                                                                   num_train_tasks = num_train_tasks, 
                                                                                   device = device, 
                                                                                   fsl_loss_fn = fsl_loss_fn, 
                                                                                   cpl_loss_fn = cpl_loss_fn, 
                                                                                   l_param = l_param,
                                                                                   project_prototypes = project_prototypes, 
                                                                                   normalize_prototypes = normalize_prototypes, 
                                                                                   n_classes = n_classes, 
                                                                                   k_support = k_support, 
                                                                                   k_query = k_query, feat_extractor=feat_extractor)

        validation_accuracy, validation_accuracy_std = evaluate_single_segment(model = model, 
                                                                               dataset = validation_dataset, 
                                                                               num_val_tasks = num_val_tasks, 
                                                                               device = device, 
                                                                               n_classes = n_classes, 
                                                                               k_support = k_support, 
                                                                               k_query = k_query, feat_extractor= feat_extractor)
        ear_stopping(val_accuracy=validation_accuracy, model=model, epoch=epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break

        train_scheduler.step()
    saved_model_pt_path = os.path.join(PROJECT_PATH, "experiments", results_path, "model.pt")
    model.load_state_dict(torch.load(saved_model_pt_path))
    trained_model = model

    return trained_model

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


def evaluate_multisegment_loop(test_dataset, n_classes, k_support, k_query, num_test_tasks, trained_model, device,
                              tie_strategy, feat_extractor):
    list_of_accuracies = []
    for i in range(num_test_tasks):
        ## Generate a test episode:
        support_list, support_labels, query_list, query_labels, audio_ids = sample_episode(
            dataset=test_dataset, n_classes=n_classes, k_support=k_support, k_query=k_query, is_test = True, device = device , feat_extractor = feat_extractor)
        support_set = [tensor.to(device) for tensor in support_list]
        query_set = [tensor.to(device) for tensor in query_list]
        support_labels = support_labels.to(device)
        query_labels = query_labels.to(device)
        audio_ids = audio_ids.to(device)
        trained_model.process_support_set(support_set, support_labels)
        with torch.no_grad():
            predictions = trained_model(query_set, inference=True)
            predicted_labels = torch.max(predictions, 1)[1]
            posterior_values = torch.max(predictions, 1)[0]
            task_accuracy = calculate_majority_vote_accuracy(predicted_labels=predicted_labels,
                                                             spectrogram_ids=audio_ids,
                                                             query_labels=query_labels,
                                                             tie_strategy=tie_strategy,
                                                             posterior_values=posterior_values)
            list_of_accuracies.append(task_accuracy)
    mean_accuracy = np.mean(list_of_accuracies)
    std_accuracy = np.std(list_of_accuracies)

    msg = {"mean_accuracy": mean_accuracy, "accuracy_std": std_accuracy}
    return msg
