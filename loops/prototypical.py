import os
import sys
from typing import Optional, Tuple

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)

from statistics import mean
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Optional
import numpy as np
from callbacks.early_stopping import EarlyStopping


def training_epoch(model, data_loader: DataLoader, optimizer: Optimizer, device, loss_function):
    all_loss = []
    model.train()
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
            classification_scores = model(query_images.to(device))

            loss = loss_function(classification_scores, query_labels.to(device))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return {"loss": mean(all_loss)}


def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    model.process_support_set(support_images, support_labels)
    with torch.no_grad():
        predictions = model(query_images, inference=True)
    if model.relation_head: 
        relation_scores = predictions.squeeze()
        print("relation_scores",relation_scores.shape)
        predicted_classes = torch.argmax(relation_scores, dim=1)
        print("predicted_classes",predicted_classes)
    number_of_correct_predictions = ((torch.max(predictions, 1)[1] == query_labels).sum().item())
    print(query_labels)
    print(number_of_correct_predictions)

    return number_of_correct_predictions, len(query_labels)


def evaluate(
    model,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> tuple[float, float]:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        mean classification accuracy and standard deviation of accuracies
    """
    # List to store accuracies for each task
    accuracies = []

    model.eval()
    with torch.no_grad():
        with tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                disable=not use_tqdm,
                desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
            ) in tqdm_eval:
                correct, total = evaluate_on_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )

                # Calculate accuracy for this task and store it
                task_accuracy = correct / total
                accuracies.append(task_accuracy)

                # Log average accuracy in real time
                tqdm_eval.set_postfix(accuracy=np.mean(accuracies))

    # Calculate mean and standard deviation of accuracies
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return mean_accuracy, std_accuracy


def prototypical_training_loop(model, training_loader, validation_loader, optimizer, device, loss_function, epochs,
                               patience, train_scheduler, experiment_folder):
    best_validation_accuracy = 0
    ear_stopping = EarlyStopping(patience=patience,
                                 verbose=True,
                                 path=os.path.join(PROJECT_PATH, "experiments", experiment_folder, "model.pt"))

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch:03}/{epochs+1:03}")
        average_training_loss = training_epoch(model=model,
                                               data_loader=training_loader,
                                               optimizer=optimizer,
                                               device=device,
                                               loss_function=loss_function)

        validation_accuracy,accuracy_std = evaluate(model=model, data_loader=validation_loader, device=device)
        ear_stopping(val_accuracy= validation_accuracy, model=model, epoch=epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break

        train_scheduler.step()

    return model


def prototypical_testing_loop(trained_model, testing_loader, device):
    test_accuracy, accuracy_std = evaluate(model=trained_model, data_loader=testing_loader, device=device)
    return {"test_accuracy": test_accuracy, "test_accuracy_std": accuracy_std}
