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
from callbacks.early_stopping import EarlyStopping_val
import torch.nn.functional as F


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
            fsl_loss = fsl_loss_fn(model.prototypes, query_features, query_labels.to(device))
            cpl_query_features, prototypes = model.contrastive_forward(project_prototypes)
            if project_prototypes == True:
                normalize_prototypes = False
            if normalize_prototypes == True:
                prototypes = F.normalize(prototypes, p=2.0, dim=1, eps=1e-12, out=None)
            print("prototypes", torch.norm(prototypes, p=2, dim=1))
            print("cpl_query_features", torch.norm(cpl_query_features, p=2, dim=1))
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

    ear_stopping = EarlyStopping_val(path=os.path.join(PROJECT_PATH, "experiments", results_path, "model.pt"),
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
