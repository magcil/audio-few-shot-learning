import torch
from datasets.datasets import MetaAudioDataset
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from models.prototypical import ContrastivePrototypicalNetworksWithoutAttention
from models.main_modules import EncoderModule,SelfAttention, ProjectionHead
import pandas as pd
from collections import Counter

def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
device = torch.device("cuda:1")
set_seed(42)
with open("experiment_config.json", "r") as f:
    experiment_config = json.load(f)

with open("models/model_params.json", "r") as f:
    model_config = json.load(f)

esc = MetaAudioDataset(experiment_config = experiment_config,root= '/data/FSD2018', split='test')
selected_labels = ['Laughter', 'Shatter', 'Knock', 'Bark', 'Keys_jangling']
#selected_labels = ['vacuum_cleaner', 'thunderstorm', 'engine', 'airplane','cow']
filtered_df = esc.data_df[esc.data_df['label'].isin(selected_labels)]
filtered_indices = filtered_df.index_column.tolist()
esc.data_df = esc.data_df[esc.data_df['index_column'].isin(filtered_indices)]



df_5_samples = (
    esc.data_df.groupby("label", group_keys=False)
    .apply(lambda x: x.sample(n=5, random_state=42))
    .reset_index(drop=True)
)
df_5_samples =  df_5_samples.sort_values(by='label')

df_remaining = esc.data_df[~esc.data_df["filename"].isin(df_5_samples["filename"])].reset_index(drop=True)
df_remaining = df_remaining.sort_values(by = 'label')

esc_support = MetaAudioDataset(experiment_config = experiment_config,root= '/data/FSD2018', split='test')
esc_query = MetaAudioDataset(experiment_config = experiment_config,root = '/data/FSD2018', split = 'test' )

esc_support.data_df = df_5_samples
esc_support.spectrograms = df_5_samples.filepath.tolist()
esc_support.class_names = df_5_samples.label.unique()
esc_support.class_to_label = {v: k for k, v in enumerate(esc_support.class_names)}
esc_support.labels = esc_support.get_labels()  

esc_query.data_df = df_remaining 
esc_query.spectograms = df_remaining.filepath.tolist()
esc_query.class_names = df_remaining.label.unique()
esc_query.class_to_label = {v: k for k, v in enumerate(esc_query.class_names)}
esc_query.labels = esc_query.get_labels()  

def create_initial_support(esc_support):
    support_labels = []
    support_specs = []
    for i in range(len(esc_support)):
        support_labels.append(esc_support[i][1])
        support_specs.append(esc_support[i][0])
    support_labels = torch.tensor(support_labels)
    support_specs = torch.cat(support_specs, dim = 0)
    return support_specs, support_labels

def get_indices(lst, iter):
    result = [iter]
    for i in range(iter, len(lst) - 1):
        if lst[i] != lst[i + 1]:  
            result.append(i + 1 + iter) 
    return result

def get_a_spec_per_label(query_dataset,iter):
    query_spec_list = []
    query_labels = []
    for i in range(len(query_dataset)):
        query_spec_list.append(query_dataset[i][0])
        query_labels.append(query_dataset[i][1])
    batch_indices = get_indices(query_labels, iter)
    query_specs = torch.cat([query_spec_list[x] for x in batch_indices],dim = 0)
    query_labels = torch.tensor([query_labels[x] for x in batch_indices])
    return query_specs, query_labels

def knn_classification(support_tensor, support_labels, query_tensor, k=3):
    """
    Classifies each query sample based on the k-nearest support neighbors (majority vote).
    
    Parameters:
        support_tensor: Tensor of shape (batch_size_support, D) 
        support_labels: Tensor of shape (batch_size_support,)
        query_tensor: Tensor of shape (batch_size_query, D)
        k: Number of nearest neighbors to consider
    
    Returns:
        predicted_labels: Tensor of shape (batch_size_query,)
    """
    # Compute pairwise Euclidean distances between query and support tensors
    distances = torch.cdist(query_tensor, support_tensor, p=2)  # Shape: (batch_size_query, batch_size_support)
    
    # Get indices of the k nearest neighbors for each query
    knn_indices = distances.topk(k, largest=False).indices  # Shape: (batch_size_query, k)

    # Gather corresponding labels from support_labels
    knn_labels = support_labels[knn_indices]  # Shape: (batch_size_query, k)
    
    # Perform majority voting
    predicted_labels = []
    for labels in knn_labels:
        most_common_label = Counter(labels.tolist()).most_common(1)[0][0]
        predicted_labels.append(most_common_label)
    
    return torch.tensor(predicted_labels)

def forward_pass(model, esc_support, esc_query,iters,update,prediction_mode, k, device = device):
    support_specs,support_labels = create_initial_support(esc_support)
    support_specs= support_specs.to(device)
    support_labels = support_labels.to(device)
    with torch.no_grad():
        model.eval()
        accuracy_scores = []

        for iter in range(iters):
            query_specs,query_labels = get_a_spec_per_label(query_dataset = esc_query, iter = iter)
            query_specs = query_specs.to(device)
            query_labels = query_labels.to(device)
            model = model.to(device)
            ## Calculate Prototypes
            if prediction_mode == "prototypical":
                model.process_support_set([support_specs], support_labels)
                predictions = model([query_specs], inference=True)
                predicted_query_labels = torch.max(predictions,1)[1]
                
            elif prediction_mode == "knn":
                support_features = model.compute_features([support_specs])
                query_features = model.compute_features([query_specs])
                predicted_query_labels = knn_classification(support_tensor = support_features, query_tensor = query_features, support_labels= support_labels, k = k)
                predicted_query_labels = predicted_query_labels.to(device)
            number_of_correct_predictions = ((predicted_query_labels == query_labels).sum().item())
            accuracy_score = number_of_correct_predictions/len(query_specs)
            accuracy_scores.append(accuracy_score)
            if update == True:
                support_specs = torch.cat([support_specs,query_specs], dim = 0)
                support_labels = torch.cat([support_labels,predicted_query_labels], dim = 0)
                
    return accuracy_scores



backbone = EncoderModule(experiment_config=experiment_config, model_config=model_config)
attention = SelfAttention(model_config=model_config)
projection = ProjectionHead(model_config=model_config)
few_shot_model = ContrastivePrototypicalNetworksWithoutAttention(backbone = backbone, projection_head = projection)
pt_file_path = "/home/csgouros/audio-few-shot-learning/experiments/FSD_PLAIN_PROTO/model.pt"
few_shot_model.load_state_dict(torch.load(pt_file_path))

k = 1

accuracy_scores_incr = forward_pass(few_shot_model, esc_support = esc_support, esc_query = esc_query, iters = 162, update = True, prediction_mode = 'prototypical', k = k)
accuracy_scores_non_incr = forward_pass(few_shot_model, esc_support = esc_support, esc_query = esc_query, iters = 162, update = False, prediction_mode = 'prototypical', k = k)
print(f"Mean Incremental Accuracy :{np.mean(accuracy_scores_incr)}-------------- Mean Non Incremental Accuracy: {np.mean(accuracy_scores_non_incr)}")


# Generate x-axis values (iterations)
iters = np.arange(1, 163)  # Assuming 35 iterations

def moving_average(data, window_size=8):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean()

# Compute moving averages for both accuracy scores
smoothed_incr = moving_average(accuracy_scores_incr, window_size=5)
smoothed_non_incr = moving_average(accuracy_scores_non_incr, window_size=5)

# Plot original data
plt.figure(figsize=(8, 5))
plt.plot(iters, accuracy_scores_incr, label="Incremental Update", marker='o', alpha=0.5)
plt.plot(iters, accuracy_scores_non_incr, label="Non-Incremental Update", marker='s', alpha=0.5)

# Plot smoothed lines
plt.plot(iters, smoothed_incr, label="Smoothed Incremental", linestyle='-', linewidth=2, color='blue')
plt.plot(iters, smoothed_non_incr, label="Smoothed Non-Incremental", linestyle='-', linewidth=2, color='red')

# Labels and title
plt.xlabel("Iterations")
plt.ylabel("Accuracy Score")
plt.title(f"Accuracy Scores over Iterations with Trend for k = {k}")
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig("incremental_plain_prototypical.jpg")

