import pdb
from copy import deepcopy
import os

intes = ['none', 'sort', 'sage', 'set2set', 'mean', 'max']
completed = {}
metrics = {}

for _intergraph in intes:
    file_name = f"output-{_intergraph}.txt"
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            content = file.read()
        groups = content.split("Group by ")[1:]
        for group in groups:
            intergraph = group.split('"intergraph": "')[1].split('"')[0]
            dataset = group.split(", Test number:")[0]
            if intergraph not in completed.keys():
                completed[intergraph] = {}
                metrics[intergraph] = {}
            if dataset not in completed[intergraph].keys():
                completed[intergraph][dataset] = {}
                metrics[intergraph][dataset] = []
            epochs = group.split("Epoch: ")[1:]
            last_epoch_index = len(epochs) - 1
            if last_epoch_index > 0:
                completed[intergraph][dataset] = last_epoch_index
            else:
                completed[intergraph][dataset] = 'not started yet'
            if 'Test auc: ' in group:
                # pdb.set_trace()
                for auc_val in group.split('Test auc: ')[1:]:
                    metrics[intergraph][dataset].append('Test auc: ' + auc_val.split("\n")[0])

for intergraph in intes:
    print(f"Intergraph: {intergraph}")
    if intergraph in completed.keys():
        for dataset in completed[intergraph].keys():
            print(f"     {dataset} -  Step: {completed[intergraph][dataset]}")
            if len(metrics[intergraph][dataset]) > 0:
               for auc_val in metrics[intergraph][dataset]:
                print(f"       Test line: {auc_val}") 
    else:
        print(f"     Not Started yet")
    print("")