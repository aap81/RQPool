import time
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch_geometric.loader import DataLoader

import utils
import pdb
import model
from name import *
import lossfunc
import traceback


def execute(args):
    data = args.data
    lr = args.lr
    batchsize = args.batchsize
    nepoch = args.nepoch
    hdim = args.hdim
    width = args.width
    depth = args.depth
    dropout = args.dropout
    normalize = args.normalize
    beta = args.beta
    gamma = args.gamma
    decay = args.decay
    seed = args.seed
    patience = args.patience
    intergraph = args.intergraph
    output_file = args.logfile
    metrics_file = args.metricsfile
    enable_print = args.enableprint
    sortk = args.sortk

    try:
        nclass = 2

        utils.set_seed(seed)

        utils.log_print("\n[NEW TEST]  - Model info:", enable_print, output_file)
        utils.log_print(json.dumps(args.__dict__, indent='\t'), enable_print, output_file)

        if 'ogbg' in data:
            graphs, adjs, features, graphlabels, train_index, val_index, test_index = utils.load_ogb_dataset(data)
        else:
            graphs, adjs, features, graphlabels, train_index, val_index, test_index = utils.load_dataset(data)
        # adjs, features, graphlabels, train_index, val_index, test_index = utils.load_data(data)
        featuredim = features[0].shape[1]

        # x_train contains all of the indices of graphs part of the training set, similarly for test and val
        # so we need to seperate the adj, features and labels based on that
        adj_train = [adjs[i] for i in train_index]
        feats_train = [features[i] for i in train_index]
        label_train = [graphlabels[i] for i in train_index]

        adj_val = [adjs[i] for i in val_index]
        feats_val = [features[i] for i in val_index]
        label_val = [graphlabels[i] for i in val_index]

        adj_test = [adjs[i] for i in test_index]
        feats_test = [features[i] for i in test_index]
        label_test = [graphlabels[i] for i in test_index]


        graphs_train = [graphs[i] for i in train_index]
        graphs_test = [graphs[i] for i in test_index]
        graphs_val = [graphs[i] for i in val_index]

        ny_0 = label_train.count(0)
        ny_1 = label_train.count(1)

        # metrics acc variables
        times = []
        
        testing = {
            'by_auc': {'idx': -1, 'auc': -1, 'f1': -1, 'accuracy': -1, 'precision': -1, 'recall': -1},
            'by_f1': {'idx': -1, 'auc': -1, 'f1': -1, 'accuracy': -1, 'precision': -1, 'recall': -1}
        }


        gad = None
        if intergraph == 'none':
            gad = model.RQGNN(featuredim, hdim, nclass, width, depth, dropout, normalize)
        else:
            gad = model.EnhancedRQGNN(featuredim, hdim, nclass, width, depth, dropout, normalize, embedding_dim=128, inter_graph_pooling=intergraph, sortk=sortk)
        optimizer = optim.Adam(gad.parameters(), lr=lr, weight_decay=decay)

        bestauc = 0
        bestf1 = 0
        bestepochauc = 0
        bestepochf1 = 0
        bestmodelauc = deepcopy(gad)
        bestmodelf1 = deepcopy(gad)

        patiencecount = 0


        utils.log_print("Starts training...", enable_print,  output_file)
        for epoch in range(nepoch):
            epoch_start = time.time()
            gad.train() # set to train mode
            train_batches = utils.generate_batches(adj_train, feats_train, label_train, batchsize, True, graphs_train)
            epoch_loss = 0

            for train_batch in train_batches:
                optimizer.zero_grad() # clears old gradients
                outputs = gad(train_batch)
                loss = lossfunc.CB_loss(train_batch.label_list, outputs, [ny_0, ny_1], nclass, beta, gamma)
                loss.backward() # Backpropagate the error to compute gradients.
                optimizer.step() # Update the model parameters using the computed gradients.
                epoch_loss += loss.item()

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            times.append(epoch_time)
            utils.log_print('Epoch: {}, loss: {}, time cost: {}'.format(epoch, epoch_loss / len(train_batches), epoch_time), enable_print, output_file)

            gad.eval()
            val_batches = utils.generate_batches(adj_val, feats_val, label_val, batchsize, False, graphs_val)
            preds = torch.Tensor()
            truths = torch.Tensor()
            for i, val_batch in enumerate(val_batches):
                outputs = gad(val_batch)
                outputs = nn.functional.softmax(outputs, dim=1)
                if i == 0:
                    preds = outputs
                    truths = val_batch.label_list
                else:
                    preds = torch.cat((preds, outputs), dim=0)
                    truths = torch.cat((truths, val_batch.label_list), dim=0)

            auc_val, f1_score_val, accuracy_val, macro_precision_val, macro_recall_val = utils.compute_metrics(preds, truths)
            utils.log_print("Val auc: {}, f1: {}, accuracy: {}, precision: {}, recall: {}".format(auc_val, f1_score_val, accuracy_val, macro_precision_val, macro_recall_val), enable_print, output_file)

            if bestauc <= auc_val:
                bestauc = auc_val
                bestepochauc = epoch
                bestmodelauc = deepcopy(gad)

            if bestf1 <= f1_score_val:
                patiencecount = 0
                bestf1 = f1_score_val
                bestepochf1 = epoch
                bestmodelf1 = deepcopy(gad)
            else:
                patiencecount += 1

            if patiencecount > patience:
                break

        utils.log_print("\nUnder the condition of auc, best idx: {}".format(bestepochauc), enable_print, output_file)
        test_batches = utils.generate_batches(adj_test, feats_test, label_test, batchsize, False, graphs_test)
        preds = torch.Tensor()
        truths = torch.Tensor()
        for i, test_batch in enumerate(test_batches):
            outputs = bestmodelauc(test_batch)
            outputs = nn.functional.softmax(outputs, dim=1)
            if i == 0:
                preds = outputs
                truths = test_batch.label_list
            else:
                preds = torch.cat((preds, outputs), dim=0)
                truths = torch.cat((truths, test_batch.label_list), dim=0)

        auc_test, f1_score_test, accuracy_test, macro_precision_test, macro_recall_test = utils.compute_metrics(preds, truths)
        utils.log_print("Test auc: {}, f1: {}, accuracy: {}, precision: {}, recall: {}\n".format(auc_test, f1_score_test, accuracy_test, macro_precision_test, macro_recall_test), enable_print, output_file)
        testing['by_auc'] = {'idx': bestepochauc, 'auc': auc_test, 'f1': f1_score_test, 'accuracy': accuracy_test, 'precision': macro_precision_test, 'recall': macro_recall_test}

        utils.log_print("Under the condition of f1, best idx: {}".format(bestepochf1), enable_print, output_file)
        test_batches = utils.generate_batches(adj_test, feats_test, label_test, batchsize, False, graphs_test)
        preds = torch.Tensor()
        truths = torch.Tensor()
        for i, test_batch in enumerate(test_batches):
            outputs = bestmodelf1(test_batch)
            outputs = nn.functional.softmax(outputs, dim=1)
            if i == 0:
                preds = outputs
                truths = test_batch.label_list
            else:
                preds = torch.cat((preds, outputs), dim=0)
                truths = torch.cat((truths, test_batch.label_list), dim=0)

        auc_test, f1_score_test, accuracy_test, macro_precision_test, macro_recall_test = utils.compute_metrics(preds, truths)
        utils.log_print("Test auc: {}, f1: {}, accuracy: {}, precision: {}, recall: {}\n".format(auc_test, f1_score_test, accuracy_test, macro_precision_test, macro_recall_test), enable_print, output_file)
        testing['by_f1'] = {'idx': bestepochf1, 'auc': auc_test, 'f1': f1_score_test, 'accuracy': accuracy_test, 'precision': macro_precision_test, 'recall': macro_recall_test}
        utils.metrics_print(f"Dataset: {data}, Intergraph: {intergraph}, avg_training_time: {sum(times)/len(times)}, testing_vals: {testing} \n", metrics_file)
    except Exception as e:
        error_traceback = traceback.format_exc()
        utils.log_print(f"An error occurred: {error_traceback}", enable_print, output_file)
