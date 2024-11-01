import os 
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from datasets.task_sampler import TaskSampler
from datasets.datasets import MetaAudioDataset
from torch.utils.data import DataLoader
from models.main_modules import EncoderModule,SelfAttention,ProjectionHead
from models.prototypical import ContrastivePrototypicalNetworks
from loops.contrastive import contrastive_testing_loop, contrastive_training_loop
from torch.optim.lr_scheduler import MultiStepLR
from loops.loss import FSL_Loss, CPL_Loss
import json
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_config", help="Path to Experiment configuration file.", required=True)
    parser.add_argument("-m", "--model_config", help="Path to model_params file", required = True)
    return parser.parse_args()


if __name__ == "__main__":
    ## Read config files
    args = parse_args()
    with open(args.experiment_config, "r") as f:
        experiment_config = json.load(f)

    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    
    ## Choosing the dataset
    data_root = '/data'
    dataset_name = experiment_config['dataset_name']
    dataset_path = data_root + "/" + dataset_name

    ## Choose device
    device_name = experiment_config['device']
    gpu_index = experiment_config['gpu_index']
    if device_name == 'cpu':
        device = device_name
    elif device_name == 'cuda':
        device = device_name + f":{gpu_index}"

    ## Choose training specifics
    n_way = experiment_config['n_way']
    n_shot = experiment_config['n_shot']
    n_query = experiment_config['n_query']
    epochs = experiment_config['num_epochs']
    n_training_tasks = experiment_config['n_training_tasks']
    n_testing_tasks = experiment_config['n_testing_tasks']
    lr = experiment_config['lr']
    l_param = experiment_config['l_param']
    m_param = experiment_config['m_param']
    t_param = experiment_config['t_param']

    print(f"Loading Dataset:::  {dataset_name}, Device used:::  {device}")

    train_set = MetaAudioDataset(root = dataset_path, split = 'train')
    val_set = MetaAudioDataset(root = dataset_path, split = 'valid')
    test_set = MetaAudioDataset(root = dataset_path, split = 'test')

    ## Initialize Samplers
    train_sampler = TaskSampler(train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_training_tasks)
    val_sampler = TaskSampler(val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_training_tasks)
    test_sampler = TaskSampler(test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_testing_tasks)

    # ## Initialize DataLoaders

    train_loader = DataLoader(train_set,
        batch_sampler=train_sampler,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,)

    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    ## Initialize Model, loss ,etc
    encoder_str = experiment_config['encoder_name']
    patience = experiment_config['patience']
    scheduler_milestones = experiment_config['scheduler_milestones']
    scheduler_gamma = experiment_config['scheduler_gamma']
    experiment_folder = experiment_config['experiment_folder']

    ## Create the experiment folder in experiments
    experiment_results_folder = f"experiments/{experiment_folder}"
    try:
        os.mkdir(f"experiments/{experiment_folder}")
    except:
        Exception("Already exists")

    backbone = EncoderModule(encoder_str = encoder_str, model_config = model_config)
    attention = SelfAttention(model_config = model_config)
    projection = ProjectionHead(model_config = model_config)
    few_shot_model = ContrastivePrototypicalNetworks(backbone = backbone, attention_model = attention, projection_head = projection).to(device)
    fsl_loss = FSL_Loss().to(device)
    cpl_loss = CPL_Loss(T = t_param, M = m_param).to(device)

    train_optimizer = torch.optim.Adam(few_shot_model.parameters(), lr = lr)
    ## Initialize scheduler
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma
    )
    print("Starting to train")
    trained_model = contrastive_training_loop(model = few_shot_model,
                            training_loader = train_loader, validation_loader = val_loader,
                            optimizer = train_optimizer, device = device, fsl_loss_fn = fsl_loss,
                            cpl_loss_fn = cpl_loss,l_param = l_param, epochs = epochs, 
                            train_scheduler = train_scheduler, patience = patience, 
                            results_path = experiment_folder)
    print(trained_model)
    print("Starting to test")

    msg =contrastive_testing_loop(trained_model = trained_model, testing_loader = test_loader, device = device)
    print(msg)
