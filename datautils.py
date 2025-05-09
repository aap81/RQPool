import time
import random
import numpy as np
import os
import torch
import urllib.request
import zipfile
import shutil
from torch_geometric.datasets import TUDataset
from name import *
import pdb
from ogb.graphproppred import PygGraphPropPredDataset # https://ogb.stanford.edu/docs/graphprop/#ogbg-molhiv
from torch_geometric.utils import to_scipy_sparse_matrix


# fixes the pattern for randomness. 
# when you set this you will always generate the same random numbers
# helpful in reproducing the results, since randomness is used to shuffle and split the dataset
# you should change the seed if you want to run the same code in a different manner
# helps in debugging if you know the right seed number
def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def get_repo_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isdir(os.path.join(current_dir, '.git')):
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find the root directory of the repository")
        current_dir = parent_dir
    return current_dir

def download_and_process_tu_dataset(datasets_path, dataset_name):
    dataset_url = f'https://www.chrsmrrs.com/graphkerneldatasets/{dataset_name}.zip'
    dataset_zip = os.path.join(datasets_path, f'{dataset_name}.zip')
    dataset_dir = os.path.join(datasets_path, dataset_name)
    raw_dir = os.path.join(dataset_dir, 'raw')

    if not os.path.exists(raw_dir):
        # Create directory if it doesn't exist
        os.makedirs(datasets_path, exist_ok=True)

        # Download the dataset
        print(f'Downloading {dataset_name} dataset...')
        urllib.request.urlretrieve(dataset_url, dataset_zip)

        # Extract the dataset
        print(f'Extracting {dataset_name} dataset...')
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(datasets_path)

        # Create the raw directory if it doesn't exist
        os.makedirs(raw_dir, exist_ok=True)

        # Move the extracted files into the raw directory
        for filename in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, filename)
            if os.path.isfile(file_path):
                shutil.move(file_path, raw_dir)

        # Clean up the zip file
        os.remove(dataset_zip)
    print(f"Processing via TUDataset {dataset_name}")
    return TUDataset(root=datasets_path, name=f'{dataset_name}')

def split_graph_dataset(graphs, datasets_path, dataset, train_ratio, test_ratio):
    print(f"\nSplitting dataset: {dataset}")
    graphlabels  = [graph.y.item() for graph in graphs]
    graphlabels = np.where(graphlabels == -1, 0, graphlabels)
    uniq_labels, counts = np.unique(graphlabels, return_counts=True)
    uniq_labels = uniq_labels.tolist()
    if len(uniq_labels) != 2:
        print(f"ERROR: Only 2 classes allowed, else select 2 classes from the list: {uniq_labels}")
        exit()

    print(f"Verifying the class distribution")
    if counts[0] < counts[1]:
        print(f" Switching the classes for better anomaly distribution")
        uniq_labels.reverse()
    if len(uniq_labels) == 2 and uniq_labels != [0, 1]:
        print(f" Converting {uniq_labels} to 0, 1")
        new_labels = []
        for i in graphlabels:
            if i == uniq_labels[0]:
                new_labels.append(0)
            else:
                new_labels.append(1)
        graphlabels = new_labels

    abnormalinds = []
    normalinds = []
    for i, label in enumerate(graphlabels):
        if label == 0:
            normalinds.append(i)
        else:
            abnormalinds.append(i)
    print(f"Class 0 - marked as normal with {len(normalinds)} graphs")
    print(f"Class 1 - marked as abnormal with {len(abnormalinds)} graphs")

    random.shuffle(normalinds)
    random.shuffle(abnormalinds)

    train_normal = np.array(normalinds[: int(train_ratio * len(normalinds))]) # first 70 percent
    val_normal = np.array(normalinds[int(train_ratio * len(normalinds)): int((1 - test_ratio) * len(normalinds))]) # 15% after the first 70%
    test_normal = np.array(normalinds[int((1 - test_ratio) * len(normalinds)): ]) # 15% after the first 70% and the later 15%

    train_abnormal = np.array(abnormalinds[: int(train_ratio * len(abnormalinds))]) # same as above
    val_abnormal = np.array(abnormalinds[int(train_ratio * len(abnormalinds)): int((1 - test_ratio) * len(abnormalinds))])
    test_abnormal = np.array(abnormalinds[int((1 - test_ratio) * len(abnormalinds)):])

    train_index = np.concatenate((train_normal, train_abnormal))
    val_index = np.concatenate((val_normal, val_abnormal))
    test_index = np.concatenate((test_normal, test_abnormal))

    random.shuffle(train_index)
    random.shuffle(val_index)
    random.shuffle(test_index)

    print("Train size: {}, normal size: {}, abnormal size: {}".format(len(train_index), len(train_normal), len(train_abnormal)))
    print("Val size: {}, normal size: {}, abnormal size: {}".format(len(val_index), len(val_normal), len(val_abnormal)))
    print("Test size: {}, normal size: {}, abnormal size: {}".format(len(test_index), len(test_normal), len(test_abnormal)))
    print("Total size: {}, generate size: {}".format(len(graphlabels), len(train_index) + len(val_index) + len(test_index)))

    train_path = os.path.join(datasets_path, dataset, "raw", dataset + TRAIN)
    val_path = os.path.join(datasets_path, dataset, "raw", dataset + VAL)
    test_path = os.path.join(datasets_path, dataset, "raw", dataset + TEST)

    np.savetxt(train_path, train_index, fmt='%d')
    np.savetxt(val_path, val_index, fmt='%d')
    np.savetxt(test_path, test_index, fmt='%d')
    np.savetxt(os.path.join(datasets_path, dataset, "raw", dataset + NEWLABEL), graphlabels, fmt='%d')

def download_and_process_ogb_dataset(dataset_name='ogbg_molhiv'):
    datasets_path = f"{get_repo_root()}/datasets"
    dataset = PygGraphPropPredDataset(name=dataset_name.replace('_', '-'), root=datasets_path)
    # Corrected label extraction
    labels = dataset.data.y.squeeze().numpy()

    if 'molpcba' in dataset_name:
        # for i in range(labels.shape[1]):
        #     valid = labels[:, i][~np.isnan(labels[:, i])]
        #     print(f"Task {i}: {np.mean(valid)}, positive samples: {np.sum(valid)}")
        task_idx = 93
        task_labels = labels[:, task_idx]

        # Remove graphs with NaN labels for this specific task
        valid_indices = ~np.isnan(task_labels)
        task_labels = task_labels[valid_indices].astype(int)
        labels = task_labels

    anomalous_class = 1
    normal_class = 0

    anomalous_indices = np.where(labels == anomalous_class)[0].tolist()
    normal_indices = np.where(labels == normal_class)[0].tolist()

    print(f"Normal graphs: {len(normal_indices)}, Anomalous graphs: {len(anomalous_indices)}")

    np.random.shuffle(anomalous_indices)
    np.random.shuffle(normal_indices)

    def split_indices(indices):
        total = len(indices)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    train_normal, val_normal, test_normal = split_indices(normal_indices)
    train_anomalous, val_anomalous, test_anomalous = split_indices(anomalous_indices)

    train_idx = np.array(train_normal + train_anomalous)
    val_idx = np.array(val_normal + val_anomalous)
    test_idx = np.array(test_normal + test_anomalous)

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    raw_dir = os.path.join(datasets_path, dataset_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    np.savetxt(os.path.join(raw_dir, f'{dataset_name}_train.txt'), train_idx, fmt='%d')
    np.savetxt(os.path.join(raw_dir, f'{dataset_name}_val.txt'), val_idx, fmt='%d')
    np.savetxt(os.path.join(raw_dir, f'{dataset_name}_test.txt'), test_idx, fmt='%d')
    np.savetxt(os.path.join(raw_dir, f'{dataset_name}_labels.txt'), labels, fmt='%d')

    return dataset, train_idx, val_idx, test_idx, labels