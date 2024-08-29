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

with open(f"logs\\hyper.txt", 'r') as file:
    content = file.read()

# Print the content
epoch_pattern = r"Epoch: (\d+), loss: ([\d\.]+), time cost: ([\d\.]+)"
val_pattern = r"Val auc: ([\d\.]+), f1: ([\d\.]+), accuracy: ([\d\.]+), precision: ([\d\.]+), recall: ([\d\.]+)"
test_pattern = r"Test auc: ([\d\.]+), f1: ([\d\.]+), accuracy: ([\d\.]+), precision: ([\d\.]+), recall: ([\d\.]+)"
dataset_groups = content.split("start hyper: ")[1:]
dataset_info = {}
hyper_pattern = r"(\w+)=([a-zA-Z0-9._'-]+)"

excel = {
    'Dataset': [],
    'hyperparameters': [],
    'AvgEpochTime seconds': [],
    "Stability - auc": [],
    "Stability - f1": [],
    "Stability - accuracy": [],
    "Stability - precision": [],
    "Stability - recall": [],
    "Peak training AUC": [],
    "Peak training F1": [],
    "Peak training Accuracy": [],
    "Peak training Precision": [],
    "Peak training recall": [],
    "AUC best - test AUC": [],
    "AUC best - test F1": [],
    "AUC best - test accuracy": [],
    "AUC best - test precision": [],
    "AUC best - test recall": [],
    "F1 best - test AUC": [],
    "F1 best - test f1": [],
    "F1 best - test accuracy": [],
    "F1 best - test precision": [],
    "F1 best - test recall": [],
}
for group in dataset_groups:
    dataset = group.split("\n")[0]
    print(f"Processing dataset: {dataset}")
    test_groups = group.split("Test number: ")[1:] # each test is what we need to check
    for test in test_groups:

        namespace_str = test.split("\n\n[NEW TEST]")[0]
        matches = re.findall(hyper_pattern, namespace_str)
        hypers = {}
        hyper_key = "{ "
        for match in matches:
            if len(hypers.keys()) != 0:
                hyper_key += " | "    
            hypers[match[0]] = match[1]
            hyper_key += f"{match[0]}: {match[1]}"
        print(f"    hypers: {hyper_key}")
        lines = test.split("\n")
        epoch_times = []
        epoch_auc = []
        epoch_f1 = []
        epoch_accuracy = []
        epoch_precision = []
        epoch_recall = []
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
        for line in lines:
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

        dataset_info[hyper_key] = {
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
            "hypers": hypers,
        }
        excel['Dataset'].append(dataset)
        excel['hyperparameters'].append(hyper_key)
        excel['AvgEpochTime seconds'].append(avg_epoch_time)
        excel["Stability - auc"].append(coefficient_of_variation(epoch_auc))
        excel["Stability - f1"].append(coefficient_of_variation(epoch_f1))
        excel["Stability - accuracy"].append(coefficient_of_variation(epoch_accuracy))
        excel["Stability - precision"].append(coefficient_of_variation(epoch_precision))
        excel["Stability - recall"].append(coefficient_of_variation(epoch_recall))
        excel["Peak training AUC"].append(max(epoch_auc))
        excel["Peak training F1"].append(max(epoch_f1))
        excel["Peak training Accuracy"].append(max(epoch_accuracy))
        excel["Peak training Precision"].append(max(epoch_precision))
        excel["Peak training recall"].append(max(epoch_recall))
        excel["AUC best - test AUC"].append(best_auc['auc'])
        excel["AUC best - test F1"].append(best_auc['f1'])
        excel["AUC best - test accuracy"].append(best_auc['accuracy'])
        excel["AUC best - test precision"].append(best_auc['precision'])
        excel["AUC best - test recall"].append(best_auc['recall'])
        excel["F1 best - test AUC"].append(best_f1['auc'])
        excel["F1 best - test f1"].append(best_f1['f1'])
        excel["F1 best - test accuracy"].append(best_f1['accuracy'])
        excel["F1 best - test precision"].append(best_f1['precision'])
        excel["F1 best - test recall"].append(best_f1['recall'])
    if None in best_auc.values() or None in best_f1.values():
        pdb.set_trace()
        raise ValueError("None found in best auc")


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

total_rows = len(excel['Dataset'])
for char in range(ord('C'), ord('W') + 1):
    lt = chr(char)
    if lt == "I":
        rule = color_scale_rule_min_red
    else:
        rule = color_scale_rule_min_green
    ws.conditional_formatting.add(f'{lt}2:{lt}{total_rows + 1}', rule)

wb.save(excel_file)
