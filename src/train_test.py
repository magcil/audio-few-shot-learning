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
from loops.loss import FSL_Loss, CPL_Loss, AngularLossClass, FSL_Loss_all_support, CPL_Loss_all_support, AngularLossClass_support
import json
import warnings
import torchaudio
from pytorch_metric_learning import losses
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
    n_way_train = experiment_config["n_way_train"]
    n_way_validation = experiment_config["n_way_validation"]
    n_way_test = experiment_config["n_way_test"]
    n_shot_train = experiment_config['n_shot_train']
    n_shot_validation = experiment_config['n_shot_validation']
    n_shot_test = experiment_config['n_shot_test']
    n_query_train = experiment_config['n_query_train']
    n_query_validation = experiment_config['n_query_validation']
    n_query_test = experiment_config['n_query_test']
    tie_strategy = experiment_config['tie_strategy']
    multi_segm = experiment_config['multi_segm']
    epochs = experiment_config['num_epochs']
    n_training_tasks = experiment_config['n_training_tasks']
    n_testing_tasks = experiment_config['n_testing_tasks']
    lr = experiment_config['lr']
    l_param = experiment_config['loss']['l_param']
    loss = experiment_config['loss']
    use_contrastive = experiment_config['use_contrastive']
    use_support_in_fsl = experiment_config['use_support_in_fsl']
    use_support_in_added = experiment_config['use_support_in_cpl']
    train_query_augmentations = experiment_config['train_query_augmentations']
    validation_query_augmentations = experiment_config['validation_query_augmentations']
    test_query_augmentations = experiment_config['test_query_augmentations']
    if loss['cpl']['use'] == True:
        m_param = loss['cpl']['m_param']
        t_param = loss['cpl']['t_param']
        if use_support_in_added:
            added_loss = CPL_Loss_all_support(T=t_param, M=m_param).to(device)
        else:
            added_loss = CPL_Loss(T=t_param, M=m_param).to(device)

    elif loss['angular']['use'] == True:
        angle = loss['angular']['angle']
        prototypes_as_anchors = loss['angular']['prototypes_as_anchors']
        if use_support_in_added:
            added_loss = AngularLossClass_support(angle = angle, prototypes_as_anchors = prototypes_as_anchors).to(device)
        else:
            added_loss = AngularLossClass(angle = angle, prototypes_as_anchors = prototypes_as_anchors).to(device)
    
    else:
        added_loss = None
        use_support_in_added = False


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
    
    for i in range(5):
        print(f"NEW RUN !!! NUMBER OF RUN ::: {i}")

        backbone = EncoderModule(experiment_config=experiment_config, model_config=model_config)
        attention = SelfAttention(model_config=model_config)
        projection = ProjectionHead(model_config=model_config)
        if experiment_config['use_attention'] == True:
            few_shot_model = ContrastivePrototypicalNetworks(backbone=backbone,
                                                        attention_model=attention,
                                                        projection_head=projection).to(device) 
        else:
            few_shot_model = ContrastivePrototypicalNetworksWithoutAttention(backbone = backbone, projection_head = projection).to(device)

        if use_support_in_fsl:
            fsl_loss = FSL_Loss_all_support().to(device)
        else:
            fsl_loss = FSL_Loss().to(device)
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
                                                n_train_classes = n_way_train, 
                                                n_validation_classes = n_way_validation, 
                                                k_support_train = n_shot_train, 
                                                k_support_validation = n_shot_validation, 
                                                k_query_train = n_query_train, 
                                                k_query_validation = n_query_validation,
                                                feat_extractor= feat_extractor,
                                                use_contrastive = use_contrastive,
                                                train_query_augmentations = train_query_augmentations,
                                                validation_query_augmentations = validation_query_augmentations,
                                                use_support_in_fsl = use_support_in_fsl,
                                                use_support_in_added = use_support_in_added)
        print(trained_model)
        print("Starting to test")
        if multi_segm == False:
            msg = evaluate_single_segment(model = trained_model, 
                                        dataset = test_set, 
                                        num_val_tasks = n_testing_tasks, 
                                        device = device, 
                                        n_classes = n_way_test, 
                                        k_support =  n_shot_test, 
                                        k_query = n_query_test, feat_extractor=feat_extractor, 
                                        eval_query_augmentation = test_query_augmentations)

        else:
            msg = evaluate_multisegment_loop(test_dataset = test_set, 
                                            n_classes = n_way_test, 
                                            k_support = n_shot_test, 
                                            k_query = n_query_test, 
                                            num_test_tasks = n_testing_tasks, 
                                            trained_model = trained_model, 
                                            device = device, 
                                            tie_strategy = tie_strategy, feat_extractor= feat_extractor,
                                            eval_query_augmentation = test_query_augmentations)

                                
        print(msg)
