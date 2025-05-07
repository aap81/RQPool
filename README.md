# RQPool

## Overview
RQPool is a framework designed for Graph-level Anomaly detection on graph datasets and running experiments using various graph pooling techniques. It supports both TUDataset and OGB datasets.

## Installation

1. First, install the required dependencies by running the following command:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scipy
pip install scikit-learn
pip install tensorflow
pip install torch
pip install torch-geometric
```bash

##  Dataset Setup
2. TUDataset only

a. To download and process any TUDataset, follow these steps:
b.Visit the TUDataset page (https://chrsmrrs.github.io/datasets/docs/datasets/) to explore the available datasets.
c. Check the name.py file for the dataset you want to use and locate the TESTING_SETS.

Run the following command to split the dataset:
```bash
python split_dataset.py
```bash
Note: If you're planning to test with an OGB dataset (e.g., ogbg_molhiv), you can skip this step.

### Running RQPool
3. To run RQPool on a dataset, use the following command:
```bash
python main.py --data ogbg_molhiv --intergraph sort --nepoch 100 --enableprint 1
```bash

You can check all available parameters in main.py