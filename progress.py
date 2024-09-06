import pdb
from copy import deepcopy
import os

intes = ['none', 'sort', 'sage', 'set2set', 'mean', 'max']
for intergraph in intes:
    file_name = f"output-{intergraph}.txt"
    completed = {}
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            content = file.read()

        groups = content.split("Group by ")[1:]
        metrics = {}
        for group in groups:
            dataset = group.split(", Test number:")[0]
            epochs = group.split("Epoch: ")[1:]
            last_epoch_index = len(epochs) - 1
            if last_epoch_index > 0:
                completed[dataset] = last_epoch_index
            else:
                completed[dataset] = 'not started yet'
            if 'Test auc: ' in group:
                metrics[dataset] = []
                # pdb.set_trace()
                for auc_val in group.split('Test auc: ')[1:]:
                    metrics[dataset].append('Test auc: ' + auc_val.split("\n")[0])

    print(f"Intergraph: {intergraph}")
    if len(completed.keys()) == 0:
        print(f"     Not Started yet")    
    for dataset in completed.keys():
        print(f"     {dataset} -  Step: {completed[dataset]}")
        if dataset in metrics.keys():
            for auc_val in metrics[dataset]:
                print(f"       Test line: {auc_val}")
    print("")

