import pdb
import re
import pandas as pd
import json
import numpy as np
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule

def coefficient_of_variation(values):
    mean = np.mean(values)
    std_dev = np.std(values)
    
    # Avoid division by zero if mean is 0
    if mean == 0:
        return float('inf')
    
    return std_dev / mean

from utils import get_repo_root

with open(f"logs/test.txt", 'r') as file:
    content = file.read()

# Print the content

epoch_pattern = r"Epoch: (\d+), loss: ([\d\.]+), time cost: ([\d\.]+)"
val_pattern = r"Val auc: ([\d\.]+), f1: ([\d\.]+), accuracy: ([\d\.]+), precision: ([\d\.]+), recall: ([\d\.]+)"
test_pattern = r"Test auc: ([\d\.]+), f1: ([\d\.]+), accuracy: ([\d\.]+), precision: ([\d\.]+), recall: ([\d\.]+)"
dataset_groups = content.split("Group by ")[1:]
hyper_pattern = r'"([^"]+)":\s*("([^"]+)"|[-+]?\d*\.?\d+),?'
dataset_info = {}

excel = {
    'Metric': [
        
    ],
    'Dataset':[],
    'none': [],
    'sort': [],
    'set2set': [],
    'sage': [],
    'mean': [],
    'max': [],
}
for group in dataset_groups:
    dataset = group.split("\n")[0]
    print(f"Processing dataset: {dataset}")
    test_groups = group.split("Test number: ")[1:] # each test is what we need to check
    intergraph_count = 0
    for _i in range(0, 21):
        excel['Dataset'].append(dataset)

    for test in test_groups:
        lines = test.split("\n")
        epoch_times = []
        epoch_auc = []
        epoch_f1 = []
        epoch_accuracy = []
        epoch_precision = []
        epoch_recall = []
        intergraph_method = None
        auc_line = True
        best_auc = {
            'auc': None,
            'f1': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
        }
        best_f1 = {
            'auc': None,
            'f1': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
        }
        hyperparameter = 'start' # 'start' or 'end'
        hyperparameter_string = ''
        for line in lines:
            if '{' in line:
                hyperparameter = 'start'
            if '}' in line:
                hyperparameter = 'end'
            if hyperparameter == 'start':
                hyper = re.search(hyper_pattern, line)
                if hyper:
                    hyperparameter_string += f" {hyper.group(1)}: {hyper.group(2)}, "
            if '\t"intergraph": ' in line:
                intergraph_count += 1
                print(f"    Intergraph: {intergraph_method}")
                intergraph_method = re.search(r'\"intergraph\": \"(.*?)\"', line).group(1)
            if 'Epoch: ' in line:
                # print(line)
                epoch_times.append(float(re.search(epoch_pattern, line).group(3)))
            if 'Val auc: ' in line:
                val_group = re.search(val_pattern, line)
                epoch_auc.append(float(val_group.group(1)))
                epoch_f1.append(float(val_group.group(2)))
                epoch_accuracy.append(float(val_group.group(3)))
                epoch_precision.append(float(val_group.group(4)))
                epoch_recall.append(float(val_group.group(5)))
            if 'Under the condition of auc' in line:
                auc_line = True
            if 'Under the condition of f1' in line:
                auc_line = False
            if 'Test auc: ' in line:
                test_group = re.search(test_pattern, line)
                if auc_line:
                    best_auc['auc'] = float(test_group.group(1))
                    best_auc['f1'] = float(test_group.group(2))
                    best_auc['accuracy'] = float(test_group.group(3))
                    best_auc['precision'] = float(test_group.group(4))
                    best_auc['recall'] = float(test_group.group(5))
                else:
                    best_f1['auc'] = float(test_group.group(1))
                    best_f1['f1'] = float(test_group.group(2))
                    best_f1['accuracy'] = float(test_group.group(3))
                    best_f1['precision'] = float(test_group.group(4))
                    best_f1['recall'] = float(test_group.group(5))
        if len(epoch_times) != 20 or len(epoch_auc) != 20 or len(epoch_f1) != 20 or len(epoch_accuracy) != 20 or len(epoch_precision) != 20 or len(epoch_recall) != 20:
            pdb.set_trace()
            raise ValueError("Epochs are incorrectly processed")
        if None in best_auc.values():
            pdb.set_trace()
            raise ValueError("None found in best auc")
        if None in best_f1.values():
            pdb.set_trace()
            raise ValueError("None found in best f1")

        avg_epoch_time = sum(epoch_times) / 20
        avg_epoch_auc = sum(epoch_auc) / 20
        avg_epoch_f1 = sum(epoch_f1) / 20
        avg_epoch_accuracy = sum(epoch_accuracy) / 20
        avg_epoch_precision = sum(epoch_precision) / 20
        avg_epoch_recall = sum(epoch_recall) / 20

        dataset_info[dataset] = {
            "intergraph_method": intergraph_method,
            "dataset": dataset,
            "epoch_times": epoch_times,
            "epoch_auc": epoch_auc,
            "epoch_f1": epoch_f1,
            "epoch_accuracy": epoch_accuracy,
            "epoch_precision": epoch_precision,
            "epoch_recall": epoch_recall,
            "avg_epoch_time": avg_epoch_time,
            "avg_epoch_auc": avg_epoch_auc,
            "avg_epoch_accuracy": avg_epoch_accuracy,
            "avg_epoch_f1": avg_epoch_f1,
            "avg_epoch_precision": avg_epoch_precision,
            "avg_epoch_recall": avg_epoch_recall,
            "best_auc": best_auc,
            "best_f1": best_f1,
            'hyperparameters': hyperparameter_string
        }
        excel[intergraph_method].append(avg_epoch_time) #'AvgEpochTime seconds',
        excel[intergraph_method].append(coefficient_of_variation(epoch_auc)) #"Stability - auc",
        excel[intergraph_method].append(coefficient_of_variation(epoch_f1)) #"Stability - f1",
        excel[intergraph_method].append(coefficient_of_variation(epoch_accuracy)) #"Stability - accuracy",
        excel[intergraph_method].append(coefficient_of_variation(epoch_precision)) #"Stability - precision",
        excel[intergraph_method].append(coefficient_of_variation(epoch_recall)) #"Stability - recall",
        excel[intergraph_method].append(max(epoch_auc)) #"Peak training AUC",
        excel[intergraph_method].append(max(epoch_f1)) #"Peak training F1",
        excel[intergraph_method].append(max(epoch_accuracy)) #"Peak training Accuracy",
        excel[intergraph_method].append(max(epoch_precision)) #"Peak training Precision",
        excel[intergraph_method].append(max(epoch_recall)) #"Peak training recall",
        excel[intergraph_method].append(best_auc['auc']) #"AUC best - test AUC",
        excel[intergraph_method].append(best_auc['f1']) #"AUC best - test F1",
        excel[intergraph_method].append(best_auc['accuracy']) #"AUC best - test accuracy",
        excel[intergraph_method].append(best_auc['precision']) #"AUC best - test precision",
        excel[intergraph_method].append(best_auc['recall']) #"AUC best - test recall",
        excel[intergraph_method].append(best_f1['auc']) #"F1 best - test AUC",
        excel[intergraph_method].append(best_f1['f1']) #"F1 best - test f1",
        excel[intergraph_method].append(best_f1['accuracy']) #"F1 best - test accuracy",
        excel[intergraph_method].append(best_f1['precision']) #"F1 best - test precision",
        excel[intergraph_method].append(best_f1['recall']) #"F1 best - test recall",
    if intergraph_count != 6:
        print(f"    {dataset} incorrect Intergraph count")
        # raise ValueError(f"{dataset} incorrect Intergraph count")
    excel['Metric'].append('AvgEpochTime seconds')
    excel['Metric'].append("Stability - auc")
    excel['Metric'].append("Stability - f1")
    excel['Metric'].append("Stability - accuracy")
    excel['Metric'].append("Stability - precision")
    excel['Metric'].append("Stability - recall")
    excel['Metric'].append("Peak training AUC")
    excel['Metric'].append("Peak training F1")
    excel['Metric'].append("Peak training Accuracy")
    excel['Metric'].append("Peak training Precision")
    excel['Metric'].append("Peak training recall")
    excel['Metric'].append("AUC best - test AUC")
    excel['Metric'].append("AUC best - test F1")
    excel['Metric'].append("AUC best - test accuracy")
    excel['Metric'].append("AUC best - test precision")
    excel['Metric'].append("AUC best - test recall")
    excel['Metric'].append("F1 best - test AUC")
    excel['Metric'].append("F1 best - test f1")
    excel['Metric'].append("F1 best - test accuracy")
    excel['Metric'].append("F1 best - test precision")
    excel['Metric'].append("F1 best - test recall")
    if None in best_auc.values():
        pdb.set_trace()
        raise ValueError("None found in best auc")


if len(excel['Metric']) != len(excel['Metric']):
    print(f"Metric column not correct {len(excel['Metric'])}")
    raise ValueError(f"Metric column not correct {len(excel['Metric'])} and metric count is {len(excel['Metric'])}")
if len(excel['none']) != len(excel['Metric']):
    print(f"none column not correct {len(excel['none'])}")
    raise ValueError(f"none column not correct {len(excel['none'])} and metric count is {len(excel['Metric'])}")
if len(excel['Dataset']) != len(excel['Metric']):
    print(f"Dataset column not correct {len(excel['Dataset'])}")
    raise ValueError(f"Dataset column not correct {len(excel['Dataset'])} and metric count is {len(excel['Metric'])}")
if len(excel['sort']) != len(excel['Metric']):
    print(f"sort column not correct {len(excel['sort'])}")
    raise ValueError(f"sort column not correct {len(excel['sort'])} and metric count is {len(excel['Metric'])}")
if len(excel['set2set']) != len(excel['Metric']):
    print(f"set2set column not correct {len(excel['set2set'])}")
    raise ValueError(f"set2set column not correct {len(excel['set2set'])} and metric count is {len(excel['Metric'])}")
if len(excel['sage']) != len(excel['Metric']):
    print(f"sage column not correct {len(excel['sage'])}")
    raise ValueError(f"sage column not correct {len(excel['sage'])} and metric count is {len(excel['Metric'])}")
if len(excel['mean']) != len(excel['Metric']):
    print(f"mean column not correct {len(excel['mean'])}")
    raise ValueError(f"mean column not correct {len(excel['mean'])} and metric count is {len(excel['Metric'])}")
if len(excel['max']) != len(excel['Metric']):
    print(f"max column not correct {len(excel['max'])}")
    raise ValueError(f"max column not correct {len(excel['max'])} and metric count is {len(excel['Metric'])}")

# with open('data.json', 'w+') as json_file:
    # json.dump(dataset_info, json_file)
# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(excel)

excel_file = 'output.xlsx'
# Save the DataFrame to an Excel file
df.to_excel(excel_file, index=False)

# Load the Excel file
wb = load_workbook(excel_file)
ws = wb.active  # Assuming we're working with the first sheet

# Define 3-color scale rule (green, yellow, red)
color_scale_rule_min_red = ColorScaleRule(
    start_type='min', start_color='FF0000',  # Red
    mid_type='percentile', mid_value=50, mid_color='FFFF00',  # Yellow
    end_type='max', end_color='00FF00'  # Green
)

color_scale_rule_min_green = ColorScaleRule(
    start_type='min', start_color='00FF00',  # Red
    mid_type='percentile', mid_value=50, mid_color='FFFF00',  # Yellow
    end_type='max', end_color='FF0000'  # Green
)


index = 0
max_i = 20
for i in range(0, len(excel['Metric'])):
    rule = None
    if index in range(0, 6):
        rule = color_scale_rule_min_green
    else:
        rule = color_scale_rule_min_red
    index += 1
    if index > max_i:
        index = 0
    ws.conditional_formatting.add(f'C{i+2}:H{i+2}', rule)

wb.save(excel_file)
