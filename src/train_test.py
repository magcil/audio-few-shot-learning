import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from datasets.datasets import MetaAudioDataset
from models.main_modules import EncoderModule, SelfAttention, ProjectionHead
from models.prototypical import ContrastivePrototypicalNetworks,ContrastivePrototypicalNetworksWithoutAttention
from loops.loops import contrastive_training_loop, evaluate_multisegment_loop,evaluate_single_segment
from torch.optim.lr_scheduler import MultiStepLR
from loops.loss import FSL_Loss, CPL_Loss, AngularLossClass
import json
import warnings
import torchaudio
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_config", help="Path to Experiment configuration file.", required=True)
    parser.add_argument("-m", "--model_config", help="Path to model_params file", required=True)
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
    tie_strategy = experiment_config['tie_strategy']
    multi_segm = experiment_config['multi_segm']
    epochs = experiment_config['num_epochs']
    n_training_tasks = experiment_config['n_training_tasks']
    n_testing_tasks = experiment_config['n_testing_tasks']
    lr = experiment_config['lr']
    l_param = experiment_config['loss']['l_param']
    loss = experiment_config['loss']
    if loss['cpl']['use'] == True:
        m_param = loss['cpl']['m_param']
        t_param = loss['cpl']['t_param']
        added_loss =  CPL_Loss(T=t_param, M=m_param).to(device)

    elif loss['angular']['use'] == True:
        angle = loss['angular']['angle']
        prototypes_as_anchors = loss['angular']['prototypes_as_anchors']
        added_loss = AngularLossClass(angle = angle, prototypes_as_anchors = prototypes_as_anchors)

    print(f"Loading Dataset:::  {dataset_name}, Device used:::  {device}")

    train_set = MetaAudioDataset(experiment_config = experiment_config,root=dataset_path, split='train')
    val_set = MetaAudioDataset(experiment_config = experiment_config, root=dataset_path, split='valid')
    test_set = MetaAudioDataset(experiment_config, root=dataset_path, split='test')

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

    backbone = EncoderModule(experiment_config=experiment_config, model_config=model_config)
    attention = SelfAttention(model_config=model_config)
    projection = ProjectionHead(model_config=model_config)
    if experiment_config['skip_attention'] == True:
        few_shot_model = ContrastivePrototypicalNetworksWithoutAttention(backbone = backbone, projection_head = projection).to(device)
    else:
        few_shot_model = ContrastivePrototypicalNetworks(backbone=backbone,
                                                     attention_model=attention,
                                                     projection_head=projection).to(device)

    fsl_loss = FSL_Loss().to(device)
    added_loss = added_loss.to(device)
    train_optimizer = torch.optim.Adam(few_shot_model.parameters(), lr=lr)
    ## Initialize scheduler
    train_scheduler = MultiStepLR(train_optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)
    print("Starting to train")
    project_prototypes = experiment_config['project_prototypes']
    normalize_prototypes = experiment_config['normalize_prototypes']
    feat_extractor  =  torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        power=2.0,
    ).to(device)


    trained_model = contrastive_training_loop(model=few_shot_model,
                                              train_dataset = train_set, 
                                              validation_dataset = val_set, 
                                              optimizer = train_optimizer, 
                                              num_train_tasks = n_training_tasks, 
                                              num_val_tasks = n_training_tasks, 
                                              device = device, 
                                              fsl_loss_fn = fsl_loss, 
                                              cpl_loss_fn = added_loss, 
                                              l_param = l_param, 
                                              epochs = epochs, 
                                              train_scheduler = train_scheduler, 
                                              patience = patience, 
                                              results_path = experiment_folder, 
                                              project_prototypes = project_prototypes, 
                                              normalize_prototypes = normalize_prototypes, 
                                              n_classes = n_way, 
                                              k_support = n_shot, 
                                              k_query = n_query, feat_extractor= feat_extractor)
    print(trained_model)
    print("Starting to test")
    if multi_segm == False:
        msg = evaluate_single_segment(model = trained_model, 
                                      dataset = test_set, 
                                      num_val_tasks = n_testing_tasks, 
                                      device = device, 
                                      n_classes = n_way, 
                                      k_support =  n_shot, 
                                      k_query = n_query, feat_extractor=feat_extractor)

    else:
        msg = evaluate_multisegment_loop(test_dataset = test_set, 
                                         n_classes = n_way, 
                                         k_support = n_shot, 
                                         k_query = n_query, 
                                         num_test_tasks = n_testing_tasks, 
                                         trained_model = trained_model, 
                                         device = device, 
                                         tie_strategy = tie_strategy, feat_extractor= feat_extractor)

                              
    print(msg)
