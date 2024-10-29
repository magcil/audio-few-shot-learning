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
        predictions = model(query_images)

    number_of_correct_predictions = ((torch.max(predictions, 1)[1] == query_labels).sum().item())

    return number_of_correct_predictions, len(query_labels)


def evaluate(
    model,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> float:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy
    """
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
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

                total_predictions += total
                correct_predictions += correct

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    return correct_predictions / total_predictions


def prototypical_training_loop(model, training_loader, validation_loader, optimizer, device, loss_function, epochs,
                               patience, train_scheduler, experiment_folder):
    best_validation_accuracy = 0
    ear_stopping = EarlyStopping(patience=patience,
                                 verbose=True,
                                 path=os.path.join(PROJECT_PATH, "logs", experiment_folder, "model.pt"))

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch:03}/{epochs+1:03}")
        average_training_loss = training_epoch(model=model,
                                               data_loader=training_loader,
                                               optimizer=optimizer,
                                               device=device,
                                               loss_function=loss_function)

        validation_accuracy = evaluate(model=model, data_loader=validation_loader, device=device)
        ear_stopping(val_loss=1 - validation_accuracy, model=model, epoch=epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break

        train_scheduler.step()

    return model


def prototypical_testing_loop(trained_model, testing_loader, device):
    test_accuracy = evaluate(model=trained_model, data_loader=testing_loader, device=device)
    return {"test_accuracy": test_accuracy}
