import pdb
import re
import pandas as pd
import json
import numpy as np
from openpyxl import load_workbook
import os
from openpyxl.formatting.rule import ColorScaleRule

# patterns
group_pattern = r"by (\w+), Test number: ([^,]+), Intergraph: (\w+)"
params_pattern = r'"([^"]+)": "([^"]+)"'                                     
epoch_pattern = r"Epoch: (\d+), loss: ([\d.]+), time cost: ([\d.]+)"
val_pattern = r"Val auc: ([\d.]+), f1: ([\d.]+), accuracy: ([\d.]+), precision: ([\d.]+), recall: ([\d.]+)"
test_pattern = r"Under the condition of (\w+), best idx: (\d+)"
test_auc_pattern = r"Test auc: ([\d.]+), f1: ([\d.]+), accuracy: ([\d.]+), precision: ([\d.]+), recall: ([\d.]+)"

by_pattern = r'\bby ([\w-]+)\b'


# pdb.set_trace()

# # file contents
# with open(f"logs/sageMeanMax.txt", 'r') as file:
#     content = file.read()


folder_path = "logs/100 epochs"
# folder_path = "logs/backup"
# folder_path = "logs/test"

data_info = {}
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Process only .txt files
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            content = file.read()

        dataset_groups = content.split("Group ")[1:]
        skipped = 0
        full_groups= 0
        for dataset_group in dataset_groups:
            lines = dataset_group.split('\n')
            
            line1 = lines[0]
            dataset = re.search(by_pattern, line1).group(1)

            group_flag = {'start': True, 'end': False }
            params_flag = {'start': False, 'end': False }
            training_flag = {'start': False, 'end': False, 'epoch_number': -1, 'loss': [], 'time': [], 'auc': [], 'f1': [], 'accuracy': [], 'precision': [], 'recall': [] }
            test_flag = {'start': False, 'count': 0, "auc": {'idx': -1} , 'f1': {'idx': -1}}
            test_number = ''
            method_str = ''
            params = {}
            test_str = ''
            method_str = ''

            for line in lines:
                if '{' in line:
                    params_flag['start'] = True
                elif '}' in line:
                    params_flag['end'] = True
                elif 'Starts training' in line:
                    training_flag['start'] = True
                elif 'Under the condition of' in line:
                    training_flag['end'] = True
                    test_flag['start'] = True
                elif len(line) < 2:
                    next
                group_match = re.search(group_pattern, line)
                params_match = re.search(params_pattern, line)
                epoch_match = re.search(epoch_pattern, line)
                val_match = re.search(val_pattern, line)
                test_match = re.search(test_pattern, line)
                test_auc_match = re.search(test_auc_pattern, line)

                
                if group_flag['start'] and not group_flag['end']:
                    if False and group_match and dataset == '':
                        print('df')
                        # dataset = group_match.group(1)
                        # test_str = group_match.group(2)
                        # method_str = group_match.group(3)
                    elif params_flag['start'] and not params_flag['end'] and params_match:
                        params[params_match.group(1)] = params_match.group(2)
                    elif training_flag['start'] and not training_flag['end'] and epoch_match:
                        training_flag['epoch_number'] = epoch_match.group(1)
                        training_flag['loss'].append(float(epoch_match.group(2)))
                        training_flag['time'].append(float(epoch_match.group(3)))
                    elif training_flag['start'] and not training_flag['end'] and val_match:
                        training_flag['auc'].append(float(val_match.group(1)))
                        training_flag['f1'].append(float(val_match.group(2)))
                        training_flag['accuracy'].append(float(val_match.group(3)))
                        training_flag['precision'].append(float(val_match.group(4)))
                        training_flag['recall'].append(float(val_match.group(5)))
                    elif training_flag['end'] and test_flag['start'] and test_match:
                        test_flag['count'] += 1             
                        test_str = test_match.group(1)
                        test_flag[test_str]['idx'] = float(test_match.group(2))
                        # print(line)
                    elif training_flag['end'] and test_flag['start'] and test_auc_match:
                        # print(line)
                        test_flag[test_str]['auc'] = float(test_auc_match.group(1))
                        test_flag[test_str]['f1'] = float(test_auc_match.group(2))
                        test_flag[test_str]['accuracy'] = float(test_auc_match.group(3))
                        test_flag[test_str]['precision'] = float(test_auc_match.group(4))
                        test_flag[test_str]['recall'] = float(test_auc_match.group(5))

            full_group = (test_flag['count'] == 2)  
            if full_group:
                epochs = len(training_flag['loss'])
                avg_time = sum(training_flag['time']) / epochs
                training_flag['avg_train'] = avg_time
                best_condition = ''
                if test_flag['auc']['auc'] > test_flag['f1']['auc']:
                    best_condition = 'auc'
                else:
                    best_condition = 'f1'
                test_flag["best_condition"] = best_condition
                # print("Best auc condition: " + best_condition)
                # print("Average traiing time:  " + str(avg_time))
                if dataset not in data_info.keys():
                    data_info[dataset] = {}
                if params['intergraph'] in data_info[dataset].keys():
                    print(f"Duplicate intergraph method in the hash {dataset} {params["intergraph"]}")
                    # pdb.set_trace()
                data_info[dataset][params['intergraph']] =  test_flag[best_condition]['auc']
                # data_info[dataset][params['intergraph']] =  avg_time
                full_groups += 1
            else:
                skipped += 1


counts = {}
for dataset in data_info.keys():
    key = ''
    max_value = 999999
    for intergraph in data_info[dataset].keys():
        if intergraph not in counts.keys():
            counts[intergraph] = 0
        if max_value > data_info[dataset][intergraph]:
        # if max_value < data_info[dataset][intergraph]:
            key = intergraph
            max_value = data_info[dataset][intergraph]
    counts[key] += 1


pdb.set_trace()
print("end")