import math

import torch
import os
import time
import random
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, csgraph
from sklearn.metrics import roc_auc_score, classification_report
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import pdb

from name import *
import batchdata
import logging
from datautils import download_and_process_ogb_dataset


def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)

def generate_batches(adjs, features, graphlabels, batchsize, shuffle, graphs=None):
    N = len(graphlabels)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    batchs = []
    for i in range(0, N, batchsize):
        ngraph = min(i + batchsize, N) - i
        nnode = sum([adjs[index[j]].shape[0] for j in range(i, min(i + batchsize, N))])
        adj_batch = lil_matrix((nnode, nnode))
        features_batch = np.zeros((nnode, features[0].shape[1]))
        label_batch = np.zeros(ngraph)
        graphpool_batch = lil_matrix((ngraph, nnode))

        xLx_batch = torch.zeros((ngraph, features[0].shape[1]))

        idx = 0

        label_count = [0, 0]
        node_belong = []
        batch_graphs = []
        for j in range(i, min(i + batchsize, N)):
            n = adjs[index[j]].shape[0]
            adj_batch[idx:idx + n, idx:idx + n] = adjs[index[j]]
            features_batch[idx:idx + n, :] = features[index[j]]
            label_batch[j - i] = graphlabels[index[j]]
            graphpool_batch[j - i, idx:idx + n] = 1
            label_count[int(graphlabels[index[j]])] += 1
            node_belong.append(list(range(idx, idx + n)))
            
            temp_L = sparse_mx_to_torch_sparse_tensor(csgraph.laplacian(adjs[index[j]], normed=True))
            temp_x = torch.FloatTensor(features[index[j]])
            xLx_batch[j - i] = torch.diag(torch.mm(torch.mm(temp_x.T, temp_L.to_dense()), temp_x))
            if graphs != None:
                batch_graphs.append(graphs[index[j]])
            idx += n

        adj_list = sparse_mx_to_torch_sparse_tensor(adj_batch)
        features_list = torch.FloatTensor(features_batch)
        label_list = torch.LongTensor(label_batch)
        graphpool_list = sparse_mx_to_torch_sparse_tensor(graphpool_batch)
        lap_list = sparse_mx_to_torch_sparse_tensor(csgraph.laplacian(adj_batch, normed=True))
        edge_index = from_scipy_sparse_matrix(adj_batch)[0]


        batchs.append(batchdata.Batch(adj_list, features_list, label_list, graphpool_list, lap_list, edge_index, label_count, node_belong, xLx_batch, batch_graphs))
    return batchs

def compute_metrics(preds, truths):

    auc = roc_auc_score(truths.detach().cpu().numpy(), preds.detach().cpu().numpy()[:, 1])

    target_names = ['C0', 'C1']
    DICT = classification_report(truths.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(axis=1), target_names=target_names, output_dict=True, zero_division = 0)

    macro_f1 = DICT['macro avg']['f1-score']

    # Extract other metrics
    accuracy = DICT['accuracy']
    macro_precision = DICT['macro avg']['precision']
    macro_recall = DICT['macro avg']['recall']

    return auc, macro_f1, accuracy, macro_precision, macro_recall

def get_repo_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isdir(os.path.join(current_dir, '.git')):
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find the root directory of the repository")
        current_dir = parent_dir
    return current_dir

def load_ogb_dataset(dataset_name="ogbg_molhiv"):
    datasets_path = f"{get_repo_root()}/datasets"
    dataset, train_idx, val_idx, test_idx, labels = download_and_process_ogb_dataset(dataset_name)

    graphs = [dataset[i] for i in range(len(dataset))]
    adjs = []
    features = []

    for graph in graphs:
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocoo()

        if graph.x is not None:
            features_matrix = graph.x.numpy()
        else:
            features_matrix = np.eye(num_nodes, 1)

        adjs.append(adj)
        features.append(features_matrix)

    return graphs, adjs, features, labels, train_idx, val_idx, test_idx

def load_dataset(dataset):
    # print(f"Loading dataset: {dataset}")

    path = os.path.join(DATADIR, dataset, "raw")
    graphlabel_path = os.path.join(path, dataset + NEWLABEL)
    graphlabels = np.loadtxt(graphlabel_path, dtype=np.int64)

    train_path = os.path.join(path, dataset + TRAIN)
    train_index = np.loadtxt(train_path, dtype=np.int64)

    val_path = os.path.join(path, dataset + VAL)
    val_index = np.loadtxt(val_path, dtype=np.int64)

    test_path = os.path.join(path, dataset + TEST)
    test_index = np.loadtxt(test_path, dtype=np.int64)

    datasets_path = f"{get_repo_root()}/datasets"

    graphs = TUDataset(root=datasets_path, name=f'{dataset}')

    adjs = []
    features = []

    for i, graph in enumerate(graphs):
        adj = to_scipy_sparse_matrix(graph.edge_index).tocoo()
        
        # Check if the graph has node features
        if graph.x is not None:
            feature_matrix = graph.x.numpy()
        else:
            num_nodes = adj.shape[0]
            feature_matrix = np.eye(num_nodes, 1)

        n_adj_nodes = adj.shape[0]
        n_feature_nodes = feature_matrix.shape[0]

        if n_adj_nodes != n_feature_nodes:
            # Identify isolated nodes
            edge_index_nodes = np.unique(graph.edge_index.numpy())
            isolated_nodes = [node for node in range(n_feature_nodes) if node not in edge_index_nodes]

            if isolated_nodes:
                # Optionally, handle isolated nodes by adding them as self-loops in the adjacency matrix
                adj_padded = lil_matrix((n_feature_nodes, n_feature_nodes))
                adj_padded[:n_adj_nodes, :n_adj_nodes] = adj

                for node_idx in isolated_nodes:
                    adj_padded[node_idx, node_idx] = 1  # self-loop for isolated nodes

                adj = adj_padded.tocoo()

        adjs.append(adj)
        features.append(feature_matrix)
    return graphs, adjs, features, graphlabels, train_index, val_index, test_index

def log_print(text, enable_print, file_name="output.txt"):
    logger = logging.getLogger("log_print")
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs
    if not logger.handlers:
        handler = logging.FileHandler(file_name)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    if enable_print:
        print(text)
    logger.info(text)

def metrics_print(text, file_name="metrics-log.txt"):
    logger = logging.getLogger("metrics_print")
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs
    if not logger.handlers:
        handler = logging.FileHandler(file_name)
        logger.addHandler(handler)

    print(text)
    logger.info(text)