import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)

from statistics import mean
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from loops.prototypical import evaluate
from callbacks.early_stopping import EarlyStopping


def training_epoch(model, data_loader: DataLoader, optimizer: Optimizer, device, fsl_loss_fn, cpl_loss_fn, l_param):
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
            cpl_query_features = model.contrastive_forward()
            cpl_loss = cpl_loss_fn(model.prototypes, cpl_query_features, query_labels.to(device))
            final_loss = fsl_loss + l_param * cpl_loss
            final_loss.backward()
            optimizer.step()
            all_loss.append(final_loss.item())
            fsl_loss_list.append(fsl_loss.item())
            cpl_loss_list.append(cpl_loss.item())
            tqdm_train.set_postfix(loss=mean(all_loss), loss_fsl=mean(fsl_loss_list), loss_cpl=mean(cpl_loss_list))

    return {"loss": mean(all_loss), "fsl_loss": mean(fsl_loss_list), "cpl_loss": mean(cpl_loss_list)}


def contrastive_training_loop(model, training_loader, validation_loader, optimizer, device, fsl_loss_fn, cpl_loss_fn,
                              l_param, epochs, train_scheduler, patience, experiment_folder):

    ear_stopping = EarlyStopping(path=os.path.join(PROJECT_PATH, "experiments", experiment_folder + "model.pt"),
                                 patience=patience,
                                 verbose=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch:03}/{epochs+1:03}")
        average_training_loss, average_fsl_loss, average_cpl_loss = training_epoch(model=model,
                                                                                   data_loader=training_loader,
                                                                                   optimizer=optimizer,
                                                                                   device=device,
                                                                                   fsl_loss_fn=fsl_loss_fn,
                                                                                   cpl_loss_fn=cpl_loss_fn,
                                                                                   l_param=l_param)

        validation_accuracy = evaluate(model=model, data_loader=validation_loader, device=device)
        ear_stopping(val_loss=1 - validation_accuracy, model=model, epoch=epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break

        train_scheduler.step()

    return model


def contrastive_testing_loop(trained_model, testing_loader, device):
    test_accuracy = evaluate(model=trained_model, data_loader=testing_loader, device=device)
    return {"test_accuracy": test_accuracy}
